
import math
from typing import Optional, Tuple, Dict

import torch, types
from torch import nn, Tensor
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
)
from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import MistralAttention

# Optional: overlap logging (only if available)
try:
    import evaluation.reuse_utils as reuse_util
except Exception:
    reuse_util = None

try:
    from evaluation.block_softmax_logger import get_block_logger
except Exception:
    get_block_logger = lambda: None


# ---------- Helpers ----------

def _rope_fast_decode_halfsplit(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.cat([xr1, xr2], dim=-1)


def _topk_chunk_mask_from_scores(
    chunk_scores: torch.Tensor,  # (B, H_kv, Q, C)
    token_budget: int,
    chunk_size: int,
    kv_len: int,
) -> torch.Tensor:
    """chunk-level top-k로 토큰 마스크 생성 (B,H_kv,Q,K)."""
    if token_budget <= 0:
        return torch.zeros(
            (chunk_scores.size(0), chunk_scores.size(1), chunk_scores.size(2), kv_len),
            dtype=torch.bool,
            device=chunk_scores.device,
        )
    num_chunks = (kv_len + chunk_size - 1) // chunk_size
    topk_c = min(max(3, token_budget // chunk_size), num_chunks)
    _, topk_idx = torch.topk(chunk_scores[..., :num_chunks], k=topk_c, dim=-1)

    token_idx = torch.arange(kv_len, device=chunk_scores.device)
    token_chunks = (token_idx // chunk_size).view(1, 1, 1, 1, kv_len)
    sel = (token_chunks == topk_idx.unsqueeze(-1)).any(dim=-2)
    return sel

def _match_k_len_replicate_last(sel_mask_pre: Tensor, k_len: int) -> Tensor:
    """(B,H,1,Kprev) → (B,H,1,Kcur): if K grows, replicate last; if shrinks, slice."""
    prev = sel_mask_pre.size(-1)
    if prev == k_len:
        return sel_mask_pre
    if prev < k_len:
        delta = k_len - prev
        tail = sel_mask_pre[..., -1:].expand(*sel_mask_pre.shape[:-1], delta)
        return torch.cat([sel_mask_pre, tail], dim=-1)
    else:
        return sel_mask_pre[..., :k_len]


def _compute_overlap(prev: torch.Tensor, curr: torch.Tensor, kv_seq_len: int) -> Dict[str, torch.Tensor]:
    """Compute overlap stats (curr_recall) between two selection masks of shape (B,H,1,K)."""
    prev_pad = _match_k_len_replicate_last(prev, int(kv_seq_len))
    inter = (prev_pad & curr).sum(dim=(-1, -2), keepdim=False)  # (B,H)
    prev_cnt = prev_pad.sum(dim=(-1, -2), keepdim=False)        # (B,H)
    curr_recall = inter.to(torch.float32) / (prev_cnt.clamp_min(1).to(torch.float32))
    return {"curr_recall": curr_recall}

def _compute_mass_coverage(p_full: torch.Tensor, sel_mask: torch.Tensor, kv_seq_len: int) -> torch.Tensor:
    """Return (B,H) coverage = sum(p_full*mask)/sum(p_full)."""
    #for compare the masks that generated in difference step
    pad_mask = _match_k_len_replicate_last(sel_mask, int(kv_seq_len))
    sel_mass = (p_full * pad_mask.float()).sum(dim=-1)                # (B,H,1)
    tot_mass = p_full.sum(dim=-1) + 1e-12                             # (B,H,1)
    return (sel_mass / tot_mass).squeeze(-1)                          # (B,H)

# --------- Chunk-max/min metadata state (kvsharemask) ---------
def _mk_ensure_state(obj) -> Dict[str, object]:
    if not hasattr(obj, "_mk_state"):
        obj._mk_state = {
            "max_list": [],  # List[Tensor(B,H_kv,C_i,D)]
            "min_list": [],  # List[Tensor(B,H_kv,C_i,D)]
            "avg_list": [],
            "pmax": None,    # Optional[Tensor(B,H_kv,1,D)]
            "pmin": None,    # Optional[Tensor(B,H_kv,1,D)]
            "pavg": None,
            "plen": 0,
            "kv_len": 0,
            # 6 MA variants for prediction tracking
            "chunk_prob_ma_g01_a02": None,
            "chunk_prob_ma_g01_a05": None,
            "chunk_prob_ma_g01_a08": None,
            "chunk_prob_ma_g02_a02": None,
            "chunk_prob_ma_g02_a05": None,
            "chunk_prob_ma_g02_a08": None,
            # cache for prediction tracking (no reuse)
            "prev_chunk_topk_indices": None,  # Previous chunk topk indices
            "prev_static_mask": None,  # Previous static chunk mask
            "last_sel_pre": None,
            # Aggregated prediction tracking (across all layers)
            "total_correct_predictions": None,  # Dict[str, int] for each MA variant
            "total_predictions": 0,
        }
    state = obj._mk_state
    return state


def _mk_reset(state: Dict[str, object]) -> None:
    state["max_list"].clear(); state["min_list"].clear(); state["avg_list"].clear()
    state["pmax"] = None; state["pmin"] = None; state["pavg"] = None
    state["chunk_prob_ma_g01_a02"] = None
    state["chunk_prob_ma_g01_a05"] = None
    state["chunk_prob_ma_g01_a08"] = None
    state["chunk_prob_ma_g02_a02"] = None
    state["chunk_prob_ma_g02_a05"] = None
    state["chunk_prob_ma_g02_a08"] = None
    state["plen"] = 0; state["kv_len"] = 0
    state["prev_chunk_topk_indices"] = None
    state["prev_static_mask"] = None
    state["last_sel_pre"] = None
    state["total_correct_predictions"] = None
    state["total_predictions"] = 0


@torch.no_grad()
def _mk_update(state: Dict[str, object], k_new: Tensor, chunk_size: int) -> None:
    if k_new is None or k_new.numel() == 0:
        return
    B, H_kv, T, D = k_new.shape
    cs = int(chunk_size)
    idx = 0

    plen: int = state["plen"]  # type: ignore
    if plen > 0:
        need = cs - plen
        take = min(need, T)
        if take > 0:
            part = k_new[:, :, idx:idx+take, :]
            pmax = part.amax(dim=2, keepdim=True)
            pmin = part.amin(dim=2, keepdim=True)
            pavg = part.mean(dim=2, keepdim=True)
            state["pmax"] = pmax if state["pmax"] is None else torch.maximum(state["pmax"], pmax)  # type: ignore
            state["pmin"] = pmin if state["pmin"] is None else torch.minimum(state["pmin"], pmin)  # type: ignore
            if state["pavg"] is None:
                state["pavg"] = pavg  # type: ignore
            else:
                prev_avg = state["pavg"]  # type: ignore
                total = plen + take
                merged = (prev_avg * plen + pavg * take) / total
                state["pavg"] = merged  # type: ignore
            state["plen"] = plen = plen + take  # type: ignore
            idx += take
        if plen == cs:
            state["max_list"].append(state["pmax"])  # type: ignore
            state["min_list"].append(state["pmin"])  # type: ignore
            state["avg_list"].append(state["pavg"])  # type: ignore
            state["pmax"] = state["pmin"] = state["pavg"] = None
            state["plen"] = 0
            plen = 0

    remain = T - idx
    if remain >= cs:
        full = remain // cs
        blk = k_new[:, :, idx:idx+full*cs, :].view(B, H_kv, full, cs, D)
        state["max_list"].append(blk.amax(dim=3))
        state["min_list"].append(blk.amin(dim=3))
        state["avg_list"].append(blk.mean(dim=3))
        idx += full * cs

    remain = T - idx
    if remain > 0:
        tail = k_new[:, :, idx:, :]
        pmax = tail.amax(dim=2, keepdim=True)
        pmin = tail.amin(dim=2, keepdim=True)
        pavg = tail.mean(dim=2, keepdim=True)
        state["pmax"] = pmax if state["pmax"] is None else torch.maximum(state["pmax"], pmax)  # type: ignore
        state["pmin"] = pmin if state["pmin"] is None else torch.minimum(state["pmin"], pmin)  # type: ignore

        prev_len = int(state["plen"])  # type: ignore
        if state["pavg"] is None:
            state["pavg"] = pavg  # type: ignore
        else:
            prev_avg = state["pavg"]  # type: ignore
            total = prev_len + remain
            merged = (prev_avg * prev_len + pavg * remain) / total
            state["pavg"] = merged  # type: ignore

        state["plen"] = prev_len + remain  # type: ignore

    state["kv_len"] = int(state["kv_len"]) + T  # type: ignore

def _mk_build_chunk_max(
    state: Dict[str, object],
    sign: Tensor,  # (B,H_kv,1,D)
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Tensor:
    """sign에 맞춘 chunk 통계 혼합 반환. 출력:(B,H_kv,C,D)"""
    B, H_kv, _, D = sign.shape
    device, dtype = sign.device, sign.dtype

    max_done = torch.cat(state["max_list"], dim=2) if state["max_list"] else None  # (B,H_kv,Cd,D)
    min_done = torch.cat(state["min_list"], dim=2) if state["min_list"] else None
    avg_done = torch.cat(state["avg_list"], dim=2) if state["avg_list"] else None

    empty = torch.empty((B, H_kv, 0, D), device=device, dtype=dtype)

    if state["plen"] > 0:
        pmax = state["pmax"]; pmin = state["pmin"]; pavg = state["pavg"]  # (B,H_kv,1,D)
        max_all = pmax if max_done is None else torch.cat([max_done, pmax], dim=2)
        min_all = pmin if min_done is None else torch.cat([min_done, pmin], dim=2)
        if pavg is not None:
            avg_all = pavg if avg_done is None else torch.cat([avg_done, pavg], dim=2)
        else:
            avg_all = avg_done if avg_done is not None else empty
    else:
        max_all = max_done if max_done is not None else empty
        min_all = min_done if min_done is not None else empty
        avg_all = avg_done if avg_done is not None else empty

    C = max_all.size(2)
    if avg_all.size(2) != C:
        avg_all = torch.zeros((B, H_kv, C, D), device=device, dtype=dtype)
    sign = sign.expand(B, H_kv, C, D)  # (B,H_kv,C,D)

    max_mix = alpha * max_all + beta * avg_all
    min_mix = alpha * min_all + beta * avg_all
    return torch.where(sign >= 0, max_mix, -min_mix)


# ---------- Hybrid Budget Allocation Helpers ----------

def _build_static_chunk_mask(
    num_chunks: int,
    static_budget_chunks: int,
    bsz: int,
    h_kv: int,
    device: torch.device,
) -> torch.Tensor:
    """Build static chunk mask with 1:3 ratio for prefix:suffix.
    Returns: (B, H_kv, num_chunks) bool mask
    """
    if static_budget_chunks <= 0 or num_chunks <= 0:
        return torch.zeros((bsz, h_kv, num_chunks), dtype=torch.bool, device=device)

    # Allocate 1:3 ratio (prefix:suffix)
    prefix_chunks = static_budget_chunks // 4
    suffix_chunks = static_budget_chunks - prefix_chunks

    # Clamp to valid range
    prefix_chunks = min(prefix_chunks, num_chunks)
    suffix_chunks = min(suffix_chunks, max(0, num_chunks - prefix_chunks))

    # Create mask
    mask = torch.zeros((bsz, h_kv, num_chunks), dtype=torch.bool, device=device)
    if prefix_chunks > 0:
        mask[:, :, :prefix_chunks] = True
    if suffix_chunks > 0:
        mask[:, :, -suffix_chunks:] = True

    return mask


def _build_hybrid_selection_mask(
    chunk_scores: torch.Tensor,  # (B, H_kv, Q, C)
    kv_seq_len: int,
    chunk_size: int,
    token_budget: int,
    static_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build hybrid selection with static + chunk topk.

    Note: The actual number of selected tokens may slightly exceed or fall short
    of `token_budget` due to chunk-based granularity. Static budget uses ceiling
    division (to ensure prefix/suffix coverage) while dynamic budget uses floor
    division, and both operate at chunk boundaries rather than exact token counts.

    Returns:
        static_mask_g: (B, H_kv, Q, K) - Static prefix/suffix selection
        topk_mask_g: (B, H_kv, Q, K) - Chunk topk selection (excluding static)
        combined_mask_g: (B, H_kv, Q, K) - Union of both
        topk_chunk_indices: (B, H_kv, Q, K_topk) - Indices of topk chunks
    """
    bsz, h_kv, q_len, num_chunks_full = chunk_scores.shape
    device = chunk_scores.device
    chunk_sz = max(1, int(chunk_size))
    num_chunks = (kv_seq_len + chunk_sz - 1) // chunk_sz

    # Validate chunk_scores shape matches computed num_chunks
    if num_chunks_full < num_chunks:
        # Pad chunk_scores with -inf for missing chunks
        pad_chunks = num_chunks - num_chunks_full
        chunk_scores = torch.nn.functional.pad(
            chunk_scores, (0, pad_chunks), value=float('-inf')
        )

    # Split budget
    static_budget_tokens = int(token_budget * static_ratio)
    chunk_topk_budget_tokens = token_budget - static_budget_tokens

    static_budget_chunks = (static_budget_tokens + chunk_sz - 1) // chunk_sz
    chunk_topk_budget_chunks = chunk_topk_budget_tokens // chunk_sz  # Remove forced minimum

    # Build static mask at chunk level
    static_chunk_mask = _build_static_chunk_mask(
        num_chunks, static_budget_chunks, bsz, h_kv, device
    )  # (B, H_kv, num_chunks)

    # Expand static mask to token level
    static_mask_tokens = static_chunk_mask.unsqueeze(-1).expand(
        bsz, h_kv, num_chunks, chunk_sz
    ).reshape(bsz, h_kv, num_chunks * chunk_sz)

    if static_mask_tokens.size(-1) > kv_seq_len:
        static_mask_tokens = static_mask_tokens[..., :kv_seq_len]
    elif static_mask_tokens.size(-1) < kv_seq_len:
        pad_len = kv_seq_len - static_mask_tokens.size(-1)
        static_mask_tokens = torch.nn.functional.pad(static_mask_tokens, (0, pad_len))

    static_mask_g = static_mask_tokens.unsqueeze(2).expand(bsz, h_kv, q_len, kv_seq_len)

    # Build chunk topk mask (excluding static chunks)
    # Zero out static chunks in scores
    chunk_scores_masked = chunk_scores.clone()
    static_chunk_mask_exp = static_chunk_mask.unsqueeze(2).expand(bsz, h_kv, q_len, num_chunks)
    chunk_scores_masked = chunk_scores_masked[..., :num_chunks]
    chunk_scores_masked[static_chunk_mask_exp] = float('-inf')

    # Select top-k from remaining chunks
    # Calculate available non-static chunks
    available_chunks = max(0, num_chunks - static_budget_chunks)
    k = min(chunk_topk_budget_chunks, available_chunks)
    # Do NOT force k=1 if no budget or no chunks remain

    if k > 0 and available_chunks > 0:
        _, topk_chunk_indices = torch.topk(chunk_scores_masked, k=k, dim=-1)  # (B, H_kv, Q, k)

        # Convert chunk indices to token mask
        token_idx = torch.arange(kv_seq_len, device=device)
        token_chunks = (token_idx // chunk_sz).view(1, 1, 1, kv_seq_len)
        topk_mask_g = (token_chunks == topk_chunk_indices.unsqueeze(-1)).any(dim=-2)
    else:
        topk_chunk_indices = torch.zeros((bsz, h_kv, q_len, 0), dtype=torch.long, device=device)
        topk_mask_g = torch.zeros((bsz, h_kv, q_len, kv_seq_len), dtype=torch.bool, device=device)

    combined_mask_g = static_mask_g | topk_mask_g

    return static_mask_g, topk_mask_g, combined_mask_g, topk_chunk_indices


def _compute_prediction_rate(
    prev_chunk_indices: Optional[torch.Tensor],  # (B, H_kv, Q, K_prev)
    curr_chunk_indices: torch.Tensor,  # (B, H_kv, Q, K_curr)
) -> float:
    """Compute how many previous chunks are still in current top-k.

    Returns: prediction rate (0.0 to 1.0)
    """
    if prev_chunk_indices is None or prev_chunk_indices.numel() == 0:
        return 0.0

    if curr_chunk_indices.numel() == 0:
        return 0.0

    # For each position, count overlap
    prev_set = prev_chunk_indices.unsqueeze(-1)  # (B, H_kv, Q, K_prev, 1)
    curr_set = curr_chunk_indices.unsqueeze(-2)  # (B, H_kv, Q, 1, K_curr)

    matches = (prev_set == curr_set).any(dim=-1)  # (B, H_kv, Q, K_prev)
    overlap_count = matches.sum(dim=-1).float()  # (B, H_kv, Q)
    prev_count = float(prev_chunk_indices.size(-1))

    if prev_count > 0:
        rate = (overlap_count / prev_count).mean().item()
    else:
        rate = 0.0

    return rate


def _compute_prediction_rate_per_head(
    prev_chunk_indices: Optional[torch.Tensor],  # (B, H_kv, Q, K_prev)
    curr_chunk_indices: torch.Tensor,  # (B, H_kv, Q, K_curr)
) -> torch.Tensor:
    """Compute prediction rate per KV head.

    Returns: (H_kv,) tensor with prediction rate for each head
    """
    if prev_chunk_indices is None or prev_chunk_indices.numel() == 0:
        return torch.zeros(curr_chunk_indices.size(1), device=curr_chunk_indices.device)

    if curr_chunk_indices.numel() == 0:
        return torch.zeros(curr_chunk_indices.size(1), device=curr_chunk_indices.device)

    # For each position, count overlap
    prev_set = prev_chunk_indices.unsqueeze(-1)  # (B, H_kv, Q, K_prev, 1)
    curr_set = curr_chunk_indices.unsqueeze(-2)  # (B, H_kv, Q, 1, K_curr)

    matches = (prev_set == curr_set).any(dim=-1)  # (B, H_kv, Q, K_prev)
    overlap_count = matches.sum(dim=-1).float()  # (B, H_kv, Q)
    prev_count = float(prev_chunk_indices.size(-1))

    if prev_count > 0:
        rate_per_head = (overlap_count / prev_count).mean(dim=(0, 2))  # Average over B and Q, keep H_kv
    else:
        rate_per_head = torch.zeros(curr_chunk_indices.size(1), device=curr_chunk_indices.device)

    return rate_per_head  # (H_kv,)


def _compute_ma_prediction_rate(
    ma_chunks: torch.Tensor,  # (B, H_kv, num_chunks)
    topk_chunk_indices: torch.Tensor,  # (B, H_kv, Q, K_topk)
    static_mask: torch.Tensor,  # (B, H_kv, num_chunks)
    bsz: int,
    num_key_value_heads: int,
    q_len: int,
) -> float:
    """Compute prediction rate for a single MA variant.

    Returns: prediction rate (0.0 to 1.0)
    """
    if ma_chunks is None or ma_chunks.numel() == 0:
        return 0.0

    if topk_chunk_indices.numel() == 0:
        return 0.0

    # Mask out static chunks
    ma_chunks_masked = ma_chunks.clone()
    ma_chunks_masked[static_mask] = float('-inf')

    # Predict top-k from MA
    k = min(topk_chunk_indices.size(-1), ma_chunks_masked.size(-1))
    if k <= 0:
        return 0.0

    ma_chunks_exp = ma_chunks_masked.unsqueeze(2).expand(bsz, num_key_value_heads, q_len, -1)
    _, ma_pred_indices = torch.topk(ma_chunks_exp, k=k, dim=-1)

    return _compute_prediction_rate(ma_pred_indices, topk_chunk_indices)


# ---------- Forward with hybrid selection ----------

def forward(self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
            ):
    bsz, q_len, _ = hidden_states.size()
    min_val = torch.finfo(hidden_states.dtype).min

    # fast path 게이트
    # if q_len > 1 or self.layer_id < 2:
    if q_len > 1 or self.layer_id < self.start_layer_id+2:
        out = self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )
        if getattr(self, "meta_data_len", 0) != 0:
            self.meta_data_len = 0
        return out

    # (B) QKV 투영
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    # (C) RoPE
    if q_len == 1:
        pos = position_ids.to(query_states.device, non_blocking=True).view(-1).to(torch.float32)
        inv = self.rotary_emb.inv_freq.to(query_states.device, non_blocking=True).to(torch.float32)
        angle = torch.outer(pos, inv)
        cos = angle.cos().to(query_states.dtype).view(-1, 1, 1, inv.numel())
        sin = angle.sin().to(query_states.dtype).view(-1, 1, 1, inv.numel())
        query_states = _rope_fast_decode_halfsplit(query_states, cos, sin)
        key_states   = _rope_fast_decode_halfsplit(key_states,   cos, sin)
    else:
        if position_ids.device != value_states.device:
            position_ids = position_ids.to(value_states.device, non_blocking=True)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # (D) KV-cache 길이/업데이트
    if isinstance(past_key_value, DynamicCache):
        kv_seq_len = past_key_value.get_seq_length()
    else:
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

    if isinstance(past_key_value, DynamicCache):
        if use_cache:
            key_states, value_states = past_key_value.update(
                key_states, value_states, layer_idx=self.layer_idx
            )
    else:
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

    # metadata update for chunk max/min
    delta = kv_seq_len - getattr(self, "meta_data_len", 0)
    state = _mk_ensure_state(self)
    if getattr(self, "meta_data_len", 0) == 0:
        _mk_reset(state)
    if delta > 0:
        k_new = key_states[:, :self.num_key_value_heads, -delta:, :]
        _mk_update(state, k_new, self.chunk_size)
    self.meta_data_len = getattr(self, "meta_data_len", 0) + delta

    # (E) 그룹화(GQA 전개)
    group_size = self.num_heads // self.num_key_value_heads  # G
    q_group = query_states.reshape(bsz, self.num_key_value_heads, group_size, q_len, self.head_dim)
    k_group = key_states.unsqueeze(2).expand(bsz, self.num_key_value_heads, group_size, kv_seq_len, self.head_dim)


    # (F) Dense QK
    q_flat = q_group.reshape(bsz * self.num_key_value_heads * group_size, q_len, self.head_dim)
    k_flat = k_group.reshape(bsz * self.num_key_value_heads * group_size, kv_seq_len, self.head_dim)
    attn_weights = torch.bmm(q_flat, k_flat.transpose(1, 2)) / math.sqrt(self.head_dim)
    attn_weights = attn_weights.reshape(bsz, self.num_key_value_heads, group_size, q_len, kv_seq_len)
    attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, kv_seq_len)


    # (G) selection 메타(sign/chunk max)
    mean_q = q_group.mean(dim=2)  # (B,H_kv,Q,D)
    sign = torch.where(mean_q.ge(0), torch.ones_like(mean_q), -torch.ones_like(mean_q))
    pos_q = mean_q * sign
    chunk_max_key = _mk_build_chunk_max(
        state,
        sign,
        alpha=float(getattr(self, "alpha", 1.0)),
        beta=float(getattr(self, "beta", 0.0)),
    )  # (B,H_kv,C,D)

    # (H) Compute chunk scores
    chunk_scores = torch.matmul(
        pos_q.to(torch.float32),                          # (B,H_kv,Q,D)
        chunk_max_key.to(torch.float32).transpose(-1, -2)     # (B,H_kv,D,C)
    )                                       # (B,H_kv,Q,C)

    # (I) 마스크 검증
    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}"
        )
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        min_val = float(torch.finfo(attn_weights.dtype).min)
        attn_weights = torch.maximum(
            attn_weights,
            torch.tensor(min_val, device=attn_weights.device, dtype=attn_weights.dtype),
        )
    else:
        min_val = float(torch.finfo(attn_weights.dtype).min)

    chunk_sz = max(1, int(self.chunk_size))
    token_budget = min(kv_seq_len, int(self.token_budget))
    static_ratio = float(getattr(self, "static_ratio", 0.0))

    # Build hybrid selection mask (static + chunk topk)
    static_mask_g, topk_mask_g, combined_mask_g, topk_chunk_indices = _build_hybrid_selection_mask(
        chunk_scores, kv_seq_len, chunk_sz, token_budget, static_ratio
    )

    # Expand to full heads (GQA)
    mask_bottom_g = combined_mask_g  # (B, H_kv, Q, K)
    sel_mask_pre = mask_bottom_g.unsqueeze(2).expand(
        bsz,
        self.num_key_value_heads,
        group_size,
        q_len,
        kv_seq_len,
    ).reshape(bsz, self.num_heads, q_len, kv_seq_len)

    # Prediction rate tracking (excluding static chunks) - 6 MA variants + direct reuse
    # Only track specific layers: 2, 8, 18, 26
    track_layers = {4, 8, 12, 16, 20}
    if getattr(self, "track_prediction", False) and topk_chunk_indices.numel() > 0 and self.layer_idx in track_layers:
        with torch.no_grad():
            num_chunks = (kv_seq_len + chunk_sz - 1) // chunk_sz

            # Build static mask for excluding prefix/suffix
            static_budget_chunks = (int(token_budget * static_ratio) + chunk_sz - 1) // chunk_sz
            static_mask = _build_static_chunk_mask(
                num_chunks, static_budget_chunks, bsz, self.num_key_value_heads,
                topk_chunk_indices.device
            )

            # Compute direct reuse rate per head from previous topk chunks
            prev_topk_indices = state.get("prev_chunk_topk_indices")
            direct_reuse_per_head = _compute_prediction_rate_per_head(prev_topk_indices, topk_chunk_indices)  # (H_kv,)

            # Define 6 MA variants (gamma, alpha pairs)
            # ma_configs = [
            #     ("chunk_prob_ma_g01_a02", 0.5, 0.2),
            #     ("chunk_prob_ma_g01_a05", 0.5, 0.5),
            #     ("chunk_prob_ma_g01_a08", 0.5, 0.8),
            #     ("chunk_prob_ma_g02_a02", 1.0, 0.2),
            #     ("chunk_prob_ma_g02_a05", 1.0, 0.5),
            #     ("chunk_prob_ma_g02_a08", 1.0, 0.8),
            # ]
            ma_configs = []

            # Build per_head_rates list
            per_head_rates = []
            for head_idx in range(self.num_key_value_heads):
                head_dict = {
                    "pred_direct": direct_reuse_per_head[head_idx].item()
                }

                # Add MA variants for this head (if any)
                for ma_key, gamma, alpha in ma_configs:
                    prev_ma = state.get(ma_key)
                    if prev_ma is not None and prev_ma.numel() > 0 and num_chunks > 0:
                        # Resize MA to match current chunk count
                        ma_chunks = prev_ma
                        if ma_chunks.size(-1) != num_chunks:
                            ma_chunks = torch.nn.functional.pad(
                                ma_chunks, (0, max(0, num_chunks - ma_chunks.size(-1)))
                            )
                            ma_chunks = ma_chunks[..., :num_chunks]

                        # Get per-head MA prediction rate
                        rate = _compute_ma_prediction_rate(
                            ma_chunks, topk_chunk_indices, static_mask,
                            bsz, self.num_key_value_heads, q_len
                        )
                    else:
                        rate = 0.0

                    # Map internal state key to prediction metric name
                    pred_key = ma_key.replace("chunk_prob_ma_", "pred_ma_")
                    pred_key = pred_key.replace("g01", "g0.1").replace("g02", "g0.2")
                    pred_key = pred_key.replace("a02", "a0.2").replace("a05", "a0.5").replace("a08", "a0.8")
                    head_dict[pred_key] = rate

                per_head_rates.append(head_dict)

            # Write to global tracker with per-head rates
            if reuse_util is not None:
                # Check write mode from global prediction state
                write_mode = getattr(reuse_util, "_PRED", None)
                write_mode = write_mode.write_mode if write_mode is not None else "accumulate"

                if write_mode == "immediate":
                    # Get task_index from state (passed from pred_reuse.py)
                    task_index = state.get("task_index", 0)
                    if hasattr(reuse_util, "pred_write_immediate"):
                        reuse_util.pred_write_immediate(
                            task_index=task_index,
                            layer_idx=self.layer_idx,
                            per_head_rates=per_head_rates,
                        )
                else:
                    # Accumulate mode (default)
                    if hasattr(reuse_util, "pred_accumulate_layer"):
                        reuse_util.pred_accumulate_layer(
                            layer_idx=self.layer_idx,
                            per_head_rates=per_head_rates,
                        )

    # Store current topk indices for next step
    state["prev_chunk_topk_indices"] = topk_chunk_indices.detach()

    # Overlap tracking
    last_sel_pre = state.get("last_sel_pre")
    if last_sel_pre is not None and getattr(self, "compute_overlap", False) and reuse_util is not None:
        with torch.no_grad():
            if self.recall_mode == "num_token":
                stats = _compute_overlap(last_sel_pre, sel_mask_pre, kv_seq_len)
                curr_recall = stats["curr_recall"].mean(dim=0)
                kv_head_recall = curr_recall.view(-1, group_size).mean(dim=1)
                reuse_util.ovp_accumulate_layer_kv(
                    layer_idx=self.layer_idx,
                    curr_recall_kv=kv_head_recall
                )
            elif self.recall_mode in ("mass", "mass_no_reuse"):
                p_full = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
                target_mask = last_sel_pre if self.recall_mode == "mass" else sel_mask_pre
                mass_cov_bh = _compute_mass_coverage(p_full, target_mask, kv_seq_len)
                kv_head_mass_cov = mass_cov_bh.mean(dim=0).view(-1, group_size).mean(dim=1)
                reuse_util.ovp_accumulate_layer_kv(
                    layer_idx=self.layer_idx,
                    curr_recall_kv=kv_head_mass_cov
                )

    state["last_sel_pre"] = sel_mask_pre

    # causal masking
    q_pos = position_ids.to(attn_weights.device).view(bsz, 1, 1)
    k_pos = torch.arange(kv_seq_len, device=attn_weights.device).view(1, 1, kv_seq_len)
    causal = (k_pos <= q_pos).to(torch.bool)                                   # (B,1,K)
    mask_bottom = sel_mask_pre & causal.unsqueeze(1)                            # (B,H,1,K)

    # attention_mask
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz,1,q_len,kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.maximum(
            attn_weights,
            torch.tensor(min_val, device=attn_weights.device, dtype=attn_weights.dtype),
        )

    # apply selection
    attn_weights.masked_fill_(~mask_bottom, min_val)
    
    # softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Update moving-average chunk probabilities (6 variants)
    if kv_seq_len > 0:
        with torch.no_grad():
            attn_per_kv = attn_weights.view(
                bsz, self.num_key_value_heads, group_size, q_len, kv_seq_len
            ).mean(dim=2)
            if q_len > 1:
                attn_per_kv = attn_per_kv.mean(dim=2)
            else:
                attn_per_kv = attn_per_kv.squeeze(2)

            num_chunks = attn_per_kv.size(-1) // chunk_sz
            if num_chunks > 0:
                usable_tokens = num_chunks * chunk_sz
                attn_trim = attn_per_kv[..., :usable_tokens]
                chunk_probs_curr = attn_trim.view(
                    bsz, self.num_key_value_heads, num_chunks, chunk_sz
                ).sum(dim=-1).to(torch.float32)

                # Update all 6 MA variants with different gamma/alpha combinations
                ma_configs = [
                    ("chunk_prob_ma_g01_a02", 0.1, 0.2),
                    ("chunk_prob_ma_g01_a05", 0.1, 0.5),
                    ("chunk_prob_ma_g01_a08", 0.1, 0.8),
                    ("chunk_prob_ma_g02_a02", 0.2, 0.2),
                    ("chunk_prob_ma_g02_a05", 0.2, 0.5),
                    ("chunk_prob_ma_g02_a08", 0.2, 0.8),
                ]

                for ma_key, gamma, alpha in ma_configs:
                    # Clamp to gamma
                    chunk_probs_clamped = torch.clamp(chunk_probs_curr, max=float(gamma))

                    # Update moving average: MA_new = alpha * MA_old + (1 - alpha) * current
                    prev_ma = state.get(ma_key)
                    if prev_ma is None or prev_ma.shape != chunk_probs_clamped.shape:
                        prev_ma = torch.zeros_like(chunk_probs_clamped)

                    updated_ma = alpha * prev_ma + (1.0 - alpha) * chunk_probs_clamped
                    state[ma_key] = updated_ma



    # Block softmax logging
    block_logger = get_block_logger() if not getattr(self, "_block_logger_failed", False) else None
    if block_logger is not None:
        if not getattr(self, "_block_logger_failed", False):
            try:
                with torch.no_grad():
                    mask_full = mask_bottom.reshape(bsz, self.num_heads, q_len, kv_seq_len)
                    attn_probs_head = torch.nn.functional.softmax(
                        attn_weights.reshape(bsz, self.num_heads, q_len, kv_seq_len),
                        dim=-1, dtype=torch.float32
                    )

                    head_filters = (
                        getattr(block_logger.config, "json_head_filters", None)
                        if hasattr(block_logger, "config") else None
                    )

                    for batch_idx in range(bsz):
                        step_val = int(position_ids[batch_idx, -1].item()) if position_ids.numel() > 0 else 0
                        q_start = int(position_ids[batch_idx, 0].item()) if position_ids.numel() > 0 else 0

                        for kv_head_idx in range(self.num_key_value_heads):
                            start = kv_head_idx * group_size
                            end = start + group_size
                            all_heads = list(range(start, end))

                            json_heads = (
                                [start] if head_filters is None else
                                [h for (layer_idx, h) in head_filters
                                 if layer_idx == self.layer_idx and start <= h < end]
                            )

                            for head_idx in all_heads:
                                probs_h = attn_probs_head[batch_idx, head_idx]
                                mask_h = mask_full[batch_idx, head_idx]
                                metadata = {"kv_seq_len": int(kv_seq_len)}

                                log_fn = block_logger.log if head_idx in json_heads else block_logger.log_pt_only
                                log_fn(
                                    layer_idx=self.layer_idx,
                                    head_idx=head_idx,
                                    kv_head_idx=kv_head_idx,
                                    step_id=step_val,
                                    chunk_size=self.chunk_size,
                                    attn_probs=probs_h,
                                    q_position_start=q_start,
                                    mask_tensor=mask_h,
                                    metadata=metadata,
                                )
            except Exception as exc:
                print(f"[block_softmax_logger] disabled due to error: {exc}")
                self._block_logger_failed = True

    # A·V + o_proj
    a_g = attn_weights.view(bsz, self.num_key_value_heads, group_size, q_len, kv_seq_len)
    attn_output = torch.einsum("bhgqk,bhkd->bhgqd", a_g, value_states).reshape(
        bsz, self.num_heads, q_len, self.head_dim
    )
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


# ---------- Helper functions ----------

def set_task_index_for_model(model, task_index: int) -> None:
    """Set task_index in all attention module states for immediate write mode."""
    for module in model.modules():
        if isinstance(module, (LlamaAttention, MistralAttention)):
            state = _mk_ensure_state(module)
            state["task_index"] = task_index


# ---------- Binding (renamed to attenion_reuse as requested) ----------

global layer_id
layer_id = 32

def enable_quest_attention_kvshare_eval_flex(model, args):
    """Bind hybrid selection forward into attention modules.
    Expects args:
        - token_budget: Total token budget
        - chunk_size: Chunk size for selection
        - static_ratio: Ratio for static prefix/suffix (0.0 to 1.0)
        - start_layer_id: Layer to start applying Quest attention
        - track_prediction: Enable prediction rate tracking (optional)
        - compute_overlap: Enable overlap tracking (optional)
        - recall_mode: Overlap mode ("num_token", "mass", "mass_no_reuse")
        - alpha, beta: Chunk max/min mixing parameters
    """
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_quest_attention_kvshare_eval_flex(module, args)

        global layer_id
        if isinstance(module, (LlamaAttention, MistralAttention)):
            layer_id -= 1
            mod = model._modules[name]
            mod.layer_id = layer_id
            mod.flash_forward = mod.forward
            mod.forward = types.MethodType(forward, mod)
            mod.token_budget = args.token_budget
            mod.chunk_size = args.chunk_size
            mod.meta_data_len = 0
            mod.static_ratio = getattr(args, "static_ratio", 0.0)
            mod.start_layer_id = getattr(args, "start_layer_id", 0)
            mod.track_prediction = getattr(args, "track_prediction", False)
            mod.compute_overlap = getattr(args, "compute_overlap", False)
            mod.recall_mode = getattr(args, "recall_mode", "num_token")
            mod.mean = getattr(args, "mean", "mean")
            mod.alpha = getattr(args, "alpha", 1.0)
            mod.beta = getattr(args, "beta", 0.0)

            # optional global logger for overlap tracking
            if mod.compute_overlap and reuse_util is not None:
                reuse_util.ovp_register_global(
                    path=getattr(args, "output_path", "attn_overlap_global.csv"),
                    model_name=getattr(args, "model", getattr(args, "model_name", "unknown")),
                    config=f"static_ratio{mod.static_ratio}",
                    budget=mod.token_budget,
                    num_kv_heads=mod.num_key_value_heads if hasattr(mod, "num_key_value_heads") else 8,
                    mode=mod.mean
                )

            # optional global logger for prediction rate tracking (per-head)
            if mod.track_prediction and reuse_util is not None:
                reuse_util.pred_register_global(
                    path=getattr(args, "pred_output_path", "prediction_rates.csv"),
                    model_name=getattr(args, "model", getattr(args, "model_name", "unknown")),
                    config=f"static_ratio{mod.static_ratio}",
                    dataset=getattr(args, "dataset", ""),
                    token_budget=mod.token_budget,
                    num_heads=mod.num_key_value_heads if hasattr(mod, "num_key_value_heads") else 8,
                    aggregation_mode="head"  # Per-head tracking
                )
