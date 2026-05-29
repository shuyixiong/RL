# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FP8 block-scaled rollout support for TRT-LLM.

Mirrors ``nemo_rl/models/generation/vllm/quantization/fp8.py`` but adapted to
TRT-LLM's PyExecutor weight-loading path:

* Trainer (Megatron policy) keeps BF16 weights.
* Inference (TRT-LLM) holds FP8 block-scaled weights (DeepSeek recipe,
  ``weight_block_size = [128, 128]``).
* At refit time we cast incoming BF16 tensors → ``(fp8_e4m3fn, scale_inv)``
  using ``cast_tensor_to_fp8_blockwise`` and inject both
  ``<name>`` and ``<name>_scale_inv`` into the dict that is then handed to
  ``ModelEngine.model_loader.reload``.  TRT-LLM's
  ``FP8BlockScalesLinearMethod.load_weights_vanilla`` picks up the
  ``weight_scale_inv`` entry and copies it into ``module.weight_scale``.

Unlike vLLM, TRT-LLM does not need monkey patches on
``process_weights_after_loading`` for the refit to preserve ``weight_loader``:
TRT-LLM's ``copy_weight``/``copy_weight_shard`` are in-place ops on the
existing ``Parameter`` buffers, so the refit-friendly invariant already holds
upstream.

All gating happens through the model's own ``QuantConfig`` (read at the
caller via :func:`is_fp8_model`); we never rely on NeMo-RL module-level
state, since TRT-LLM spawns its TP workers as separate Ray-actor processes
and any such state would be unset there.
"""
from typing import Any

import torch

# DeepSeek-style FP8 block-scale recipe; identical to the vLLM path so that
# trainers can reuse the same calibration / scale-derivation logic across both
# backends.
#
# ``modules_to_not_convert`` is HF's standard glob list of layers to keep in
# BF16.  Defaults cover the MoE router (``*mlp.gate``) and the output head
# (``lm_head``), which are precision-sensitive enough that quantising them
# typically costs convergence.  TRT-LLM merges this with its own built-in
# excludes (``*kv_b_proj*``, ``*k_b_proj*``, ``*eh_proj`` —
# see ``tensorrt_llm/_torch/model_config.py:399-406``).
FP8_BLOCK_QUANT_KWARGS: dict[str, Any] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": ["*mlp.gate", "lm_head"],
}


# ---------------------------------------------------------------------------- #
#  Public API
# ---------------------------------------------------------------------------- #


def init_fp8(
    trtllm_cfg: dict[str, Any],
    model_name: str,
    model_parallel_size: int,
) -> dict[str, Any]:
    """Validate FP8 config and produce ``LLM(...)``-bound kwargs.

    Called from :class:`TrtllmGenerationWorker` (or its async sibling) when
    ``trtllm_cfg["precision"] == "fp8"``.  The returned dict is merged into
    ``llm_kwargs`` before constructing TRT-LLM's ``LLM``/``AsyncLLM``.
    """
    kv_cache_dtype = trtllm_cfg.get("kv_cache_dtype", "auto")
    if kv_cache_dtype not in ("auto", "fp8"):
        raise ValueError(
            f"trtllm_cfg.kv_cache_dtype must be one of ['auto', 'fp8'], got "
            f"{kv_cache_dtype!r}"
        )
    # NOTE: ``kv_cache_dtype="fp8"`` is wired through ``KvCacheConfig(dtype=...)``
    # in the worker — it does NOT flow through this function's return value.
    # See ``trtllm_worker.py`` / ``trtllm_worker_async.py`` for the
    # ``KvCacheConfig`` construction site.

    # ``load_format="dummy"``: when running FP8 rollouts the trainer (Megatron,
    # BF16) authoritatively supplies weights at refit, so loading the HF
    # checkpoint from disk is wasted I/O — and worse, if the checkpoint is a
    # pre-quantised FP8 model whose layout/recipe differs from
    # ``FP8_BLOCK_QUANT_KWARGS``, the parsed weights would silently mis-align
    # with our scale schema.  Forcing ``dummy`` allocates the FP8 buffers
    # without reading from disk; the first refit fills them with real values.
    return {
        "model_kwargs": {
            "quantization_config": dict(FP8_BLOCK_QUANT_KWARGS),
        },
        "load_format": "dummy",
    }


def is_fp8_model(quant_config) -> bool:
    """Whether the live model is FP8 block-scaled.

    Reads TRT-LLM's own ``QuantConfig`` so that every TP worker sees the same
    authoritative answer regardless of which process ``init_fp8`` ran in.
    """
    if quant_config is None:
        return False
    layer_mode = getattr(quant_config, "layer_quant_mode", None)
    if layer_mode is None:
        return False
    return layer_mode.has_fp8_block_scales()


# ---------------------------------------------------------------------------- #
#  BF16 → FP8 blockwise cast (math identical to the vLLM rollout path)
# ---------------------------------------------------------------------------- #


def cast_tensor_to_fp8_blockwise(
    data_hp: torch.Tensor,
    weight_block_size: list[int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast a 2-D high-precision weight tile to FP8 with per-block scale.

    Returns ``(fp_data, descale_fp)``:

    * ``fp_data``  is ``torch.float8_e4m3fn`` of the original shape.
    * ``descale_fp`` is ``[ceil(M/bs0), ceil(N/bs1)]`` ``float32`` scale-inv
      (i.e. ``max_abs / fp8_max``) — same convention TRT-LLM's
      ``FP8BlockScalesLinearMethod`` expects on the ``weight_scale_inv`` key.
    """
    assert data_hp.dim() == 2, (
        f"cast_tensor_to_fp8_blockwise expects a 2-D weight, got shape "
        f"{tuple(data_hp.shape)}"
    )

    block_size0, block_size1 = weight_block_size
    assert block_size0 == block_size1, (
        "Only square blocks are supported (TRT-LLM FP8BlockScales uses [128,128])."
    )

    # Pad to multiples of block_size on each dim, replicating the last element
    # (this matches the vLLM cast).
    shape_before_padding = data_hp.shape
    pad0 = (
        0 if data_hp.shape[0] % block_size0 == 0
        else block_size0 - data_hp.shape[0] % block_size0
    )
    pad1 = (
        0 if data_hp.shape[1] % block_size1 == 0
        else block_size1 - data_hp.shape[1] % block_size1
    )
    if pad0 or pad1:
        data_hp = torch.nn.functional.pad(
            data_hp, (0, pad1, 0, pad0), mode="constant", value=data_hp[-1, -1]
        )

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    original_shape = data_hp.shape
    blk_m = data_hp.shape[0] // block_size0
    blk_n = data_hp.shape[1] // block_size1

    # (M, N) -> (BLK_M, BS, BLK_N, BS) -> (BLK_M, BLK_N, BS*BS)
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
    data_hp = data_hp.permute(0, 2, 1, 3).to(torch.float32).contiguous().flatten(
        start_dim=2
    )

    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)
    descale = max_abs / fp8_max  # scale_inv

    scale_fp = fp8_max / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)
    descale_fp = torch.reciprocal(scale_fp)

    data_lp = torch.clamp(data_hp * scale_fp, min=-fp8_max, max=fp8_max)
    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BS*BS) -> (M, N)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(original_shape)
    )

    # Strip padding back to original shape.
    if fp_data.shape != shape_before_padding:
        fp_data = fp_data[: shape_before_padding[0], : shape_before_padding[1]]

    return fp_data, descale_fp.squeeze(-1)


# ---------------------------------------------------------------------------- #
#  Weight injection
# ---------------------------------------------------------------------------- #


# HF-style suffixes that always land in an FP8 linear layer for the dense
# architectures we ship today (Qwen3, Llama, Mistral, Gemma).  For MoE archs
# (FusedMoE), :func:`load_weights` needs an extra branch — Phase 2 follow-up.
_FP8_LINEAR_SUFFIXES: tuple[str, ...] = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def _is_fp8_weight_by_name(name: str) -> bool:
    """Cheap heuristic: does this HF param name map to an FP8 linear layer?

    Returns False for layer-norms, embeddings, biases, and anything else that
    TRT-LLM keeps in BF16 even when the rest of the model is FP8 block-scaled.

    The heuristic suits dense decoder LLMs.  MoE expert weights have a
    different naming convention and need a separate branch — guarded against
    here so callers can extend the rule without silent regressions.
    """
    if not name.endswith(".weight"):
        return False
    # Bias / norm / embedding never go through FP8BlockScalesLinearMethod.
    if any(
        kw in name
        for kw in ("layernorm", "norm.weight", "embed", "lm_head", "bias")
    ):
        return False
    return any(name.endswith(suffix) for suffix in _FP8_LINEAR_SUFFIXES)


def load_weights(
    weight_list: list[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Convert a list of ``(name, bf16_tensor)`` into the FP8 dict TRT-LLM expects.

    For FP8 linear weights this expands ``(name, weight)`` into two entries:

    * ``name``                          → FP8 ``torch.float8_e4m3fn`` data
    * ``name`` with ``.weight`` replaced
      by ``.weight_scale_inv``          → ``float32`` block scale_inv

    Non-FP8 weights pass through unchanged.

    The caller (``trtllm_backend.NcclExtension``) gates on
    :func:`is_fp8_model` before invoking this function, so the cast is only
    reached when the live model is FP8 block-scaled.
    """
    block_size = FP8_BLOCK_QUANT_KWARGS["weight_block_size"]
    out: dict[str, torch.Tensor] = {}
    for name, tensor in weight_list:
        if not _is_fp8_weight_by_name(name):
            out[name] = tensor
            continue

        # Cast on whatever device the tensor came in on (typically CUDA after
        # the packed broadcast / IPC unpack).  cast_tensor_to_fp8_blockwise
        # uses fp32 math internally, then casts to fp8.
        fp8_data, scale_inv = cast_tensor_to_fp8_blockwise(
            tensor.to(torch.float32), weight_block_size=block_size,
        )
        scale_name = name[: -len(".weight")] + ".weight_scale_inv"
        out[name] = fp8_data
        out[scale_name] = scale_inv

    return out
