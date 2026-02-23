#!/usr/bin/env python3
"""Import HuggingFace NVFP4 compressed-tensors model into nmoe checkpoint format.

This tool converts a HuggingFace-format NVFP4 (compressed_tensors) DeepSeek V3.2
model into the nmoe checkpoint format (rd.pt + dp_rank_*.pt), preserving the NVFP4
triplet encoding intact. No decompression or re-quantization is performed.

The import pipeline:
  1. Load NVFP4 safetensors from the HF model directory
  2. Map HF weight names -> nmoe weight names (handling NVFP4 triplet suffixes)
  3. Stack individual expert NVFP4 triplets into nmoe stacked [n_local, ...] format
  4. Shard experts across EP ranks: n_routed_experts / ep_size experts per rank
  5. For each expert triplet (weight_packed, weight_scale, weight_global_scale),
     all 3 tensors go to the same EP rank
  6. Save as rd.pt (dense/router/shared + NVFP4 attention) + dp_rank_*.pt (expert shards)

NVFP4 compressed_tensors format (per quantized weight):
  - weight_packed:      [out_features, in_features // 2]  uint8   (2 FP4 per byte)
  - weight_scale:       [out_features, in_features // 16] float8_e4m3fn  (group_size=16)
  - weight_global_scale: [1]                              float32

Target model: DeepSeek-V3.2-REAP-345B-NVFP4A16-v2 (184 GB, 65 shards)
  - 128 routed experts, 8 activated, 1 shared
  - 61 layers, first 3 dense (first_k_dense_replace=3)
  - vocab_size=129280, dim=7168, moe_inter_dim=2048
  - 22,943 quantized weights (NVFP4 triplets), 485 BF16 (norms, gates, embed, lm_head)

Architecture: Option C -- NVFP4 primary weights + FP8 optimizer states + ECO
  (no BF16 master weights; ECO injects quantization error into Adam momentum)

Usage:
    # Standard import (EP=8, NVFP4 preserved):
    python -m nmoe.tools.import_from_hf \
        --model /home/nourdine/DeepSeek-V3.2-REAP-345B-NVFP4A16-v2 \
        --output /data/checkpoints/dsv3_reap_sft/iteration_00000 \
        --ep-size 8

    # Dry-run validation (no tensor loading):
    python -m nmoe.tools.import_from_hf \
        --model /home/nourdine/DeepSeek-V3.2-REAP-345B-NVFP4A16-v2 \
        --validate-only

    # Structural test (plan sharding without writing full checkpoint):
    python -m nmoe.tools.import_from_hf \
        --model /home/nourdine/DeepSeek-V3.2-REAP-345B-NVFP4A16-v2 \
        --plan-only --ep-size 8
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# REAP-345B architecture constants
REAP_N_LAYERS = 61
REAP_N_DENSE_LAYERS = 3  # first_k_dense_replace
REAP_N_ROUTED_EXPERTS = 128
REAP_N_SHARED_EXPERTS = 1
REAP_DIM = 7168
REAP_MOE_INTER_DIM = 2048
REAP_VOCAB_SIZE = 129280

# nmoe checkpoint format version (must match checkpoint.py CHECKPOINT_FORMAT_VERSION)
CHECKPOINT_FORMAT_VERSION = 3

# NVFP4 compressed_tensors triplet suffixes
NVFP4_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_global_scale")


# ---------------------------------------------------------------------------
# NVFP4 tensor classification
# ---------------------------------------------------------------------------

def strip_nvfp4_suffix(key: str) -> Tuple[Optional[str], Optional[str]]:
    """Strip NVFP4 compressed_tensors suffix from a key.

    Args:
        key: HF tensor name, e.g. 'model.layers.0.self_attn.q_a_proj.weight_packed'

    Returns:
        (base_name, suffix) if NVFP4 suffix found, e.g.
        ('model.layers.0.self_attn.q_a_proj', 'weight_packed').
        (None, None) if no NVFP4 suffix.
    """
    for sfx in NVFP4_SUFFIXES:
        if key.endswith(sfx):
            base = key[: -len(sfx)]
            suffix_name = sfx.lstrip(".")
            return base, suffix_name
    return None, None


def is_nvfp4_key(key: str) -> bool:
    """Check if a key belongs to an NVFP4 triplet."""
    return any(key.endswith(sfx) for sfx in NVFP4_SUFFIXES)


# ---------------------------------------------------------------------------
# HF -> nmoe weight name mapping (base names, without NVFP4 suffixes)
# ---------------------------------------------------------------------------

def build_hf_to_nmoe_base_mapping(
    n_layers: int = REAP_N_LAYERS,
    n_dense_layers: int = REAP_N_DENSE_LAYERS,
    n_shared_experts: int = REAP_N_SHARED_EXPERTS,
) -> Dict[str, str]:
    """Build HF base weight name -> nmoe base weight name mapping.

    For NVFP4 weights, this maps the base name (without suffix). The suffix
    (weight_packed / weight_scale / weight_global_scale) is preserved and
    appended to the nmoe name.

    For BF16 weights, the HF key ends in '.weight' or '.bias' and maps directly.

    Returns:
        Dict mapping HF base/full name -> nmoe base/full name.
    """
    mapping = {}

    # Embedding and head (BF16 -- not quantized)
    mapping["model.embed_tokens.weight"] = "embedding.weight"
    mapping["lm_head.weight"] = "lm_head.weight"
    mapping["model.norm.weight"] = "norm.weight"

    for layer_id in range(n_layers):
        hf = f"model.layers.{layer_id}"
        nm = f"blocks.{layer_id}"

        # Attention norms (BF16)
        mapping[f"{hf}.input_layernorm.weight"] = f"{nm}.attn_norm.weight"
        mapping[f"{hf}.post_attention_layernorm.weight"] = f"{nm}.ffn_norm.weight"

        # MLA attention weights (NVFP4 triplets -- map base names)
        mapping[f"{hf}.self_attn.q_a_proj"] = f"{nm}.attn.wq_a"
        mapping[f"{hf}.self_attn.q_a_layernorm.weight"] = f"{nm}.attn.q_norm.weight"
        mapping[f"{hf}.self_attn.q_b_proj"] = f"{nm}.attn.wq_b"
        mapping[f"{hf}.self_attn.kv_a_proj_with_mqa"] = f"{nm}.attn.wkv_a"
        mapping[f"{hf}.self_attn.kv_a_layernorm.weight"] = f"{nm}.attn.kv_norm.weight"
        mapping[f"{hf}.self_attn.kv_b_proj"] = f"{nm}.attn.wkv_b"
        mapping[f"{hf}.self_attn.o_proj"] = f"{nm}.attn.wo"

        # NSA Indexer weights (NVFP4 triplets for projections, BF16 for norms)
        mapping[f"{hf}.self_attn.indexer.wq_b"] = f"{nm}.attn.indexer.wq_b"
        mapping[f"{hf}.self_attn.indexer.wk"] = f"{nm}.attn.indexer.wk"
        mapping[f"{hf}.self_attn.indexer.weights_proj"] = f"{nm}.attn.indexer.weights_proj"
        mapping[f"{hf}.self_attn.indexer.k_norm.weight"] = f"{nm}.attn.indexer.k_norm.weight"
        mapping[f"{hf}.self_attn.indexer.k_norm.bias"] = f"{nm}.attn.indexer.k_norm.bias"

        is_moe_layer = layer_id >= n_dense_layers

        if is_moe_layer:
            # Router (BF16)
            mapping[f"{hf}.mlp.gate.weight"] = f"{nm}.ffn.router.router_weight"
            mapping[f"{hf}.mlp.gate.e_score_correction_bias"] = f"{nm}.ffn.router.bias"

            # Shared expert (NVFP4 triplets -- map base names)
            if n_shared_experts > 0:
                mapping[f"{hf}.mlp.shared_experts.gate_proj"] = f"{nm}.ffn._shared.w1"
                mapping[f"{hf}.mlp.shared_experts.up_proj"] = f"{nm}.ffn._shared.w3"
                mapping[f"{hf}.mlp.shared_experts.down_proj"] = f"{nm}.ffn._shared.w2"

            # Expert weights are NOT mapped here -- handled by stacking logic.
        else:
            # Dense MLP (NVFP4 triplets -- map base names)
            mapping[f"{hf}.mlp.gate_proj"] = f"{nm}.ffn.w1"
            mapping[f"{hf}.mlp.up_proj"] = f"{nm}.ffn.w3"
            mapping[f"{hf}.mlp.down_proj"] = f"{nm}.ffn.w2"

    return mapping


# ---------------------------------------------------------------------------
# Expert weight parsing for NVFP4 compressed_tensors
# ---------------------------------------------------------------------------

# Regex for expert NVFP4 triplet keys:
#   model.layers.5.mlp.experts.42.gate_proj.weight_packed
#   model.layers.5.mlp.experts.42.gate_proj.weight_scale
#   model.layers.5.mlp.experts.42.gate_proj.weight_global_scale
_EXPERT_NVFP4_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)"
    r"\.(gate|up|down)_proj"
    r"\.(weight_packed|weight_scale|weight_global_scale)"
)

_PROJ_MAP = {"gate": "W1", "up": "W3", "down": "W2"}


def parse_expert_nvfp4_key(name: str) -> Optional[Dict[str, Any]]:
    """Parse an NVFP4 expert weight key.

    Args:
        name: e.g. 'model.layers.5.mlp.experts.42.gate_proj.weight_packed'

    Returns:
        Dict with {layer_id, expert_id, proj_type ('W1'/'W3'/'W2'),
                    suffix ('weight_packed'/'weight_scale'/'weight_global_scale')}
        or None if not an expert key.
    """
    m = _EXPERT_NVFP4_RE.match(name)
    if m is None:
        return None
    return {
        "layer_id": int(m.group(1)),
        "expert_id": int(m.group(2)),
        "proj_type": _PROJ_MAP[m.group(3)],
        "suffix": m.group(4),
    }


# ---------------------------------------------------------------------------
# Safetensors loading (streaming, shard-by-shard to avoid OOM)
# ---------------------------------------------------------------------------

def load_safetensors_index(model_dir: Path) -> Dict[str, str]:
    """Load the weight_map from model.safetensors.index.json.

    Returns:
        Dict mapping weight name -> shard filename.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No model.safetensors.index.json found in {model_dir}. "
            "Is this a valid HuggingFace model directory?"
        )
    with open(index_path) as f:
        index = json.load(f)
    return index["weight_map"]


def group_weights_by_shard(weight_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Group weight names by their shard file."""
    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        shard_to_keys[shard].append(key)
    return dict(shard_to_keys)


def load_shard_tensors(
    model_dir: Path,
    shard_name: str,
    keys_to_load: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Load specific tensors from a safetensors shard.

    Args:
        model_dir: Model directory.
        shard_name: Shard filename.
        keys_to_load: If provided, only load these keys (for memory efficiency).

    Returns:
        Dict of loaded tensors.
    """
    from safetensors.torch import load_file

    shard_path = model_dir / shard_name
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    all_tensors = load_file(str(shard_path), device="cpu")

    if keys_to_load is not None:
        keys_set = set(keys_to_load)
        return {k: v for k, v in all_tensors.items() if k in keys_set}
    return all_tensors


# ---------------------------------------------------------------------------
# Expert NVFP4 triplet stacking
# ---------------------------------------------------------------------------

def stack_expert_nvfp4_triplets(
    expert_tensors: Dict[int, torch.Tensor],
    n_total_experts: int,
    proj_type: str,
    suffix: str,
) -> torch.Tensor:
    """Stack individual expert NVFP4 tensors into [n_experts, ...] format.

    For weight_packed:      HF [out_dim, in_dim//2] -> stacked [n_experts, out_dim, in_dim//2]
    For weight_scale:       HF [out_dim, in_dim//16] -> stacked [n_experts, out_dim, in_dim//16]
    For weight_global_scale: HF [1] -> stacked [n_experts, 1]

    No transpose is applied -- NVFP4 packed data is kept in HF layout (row-major,
    out_dim on rows). The dequantization logic at load time must know this layout.

    Args:
        expert_tensors: Dict mapping expert_id -> tensor.
        n_total_experts: Expected total expert count.
        proj_type: 'W1', 'W3', or 'W2'.
        suffix: 'weight_packed', 'weight_scale', or 'weight_global_scale'.

    Returns:
        Stacked tensor [n_total_experts, ...].
    """
    if len(expert_tensors) != n_total_experts:
        raise ValueError(
            f"Expected {n_total_experts} experts for {proj_type}.{suffix}, "
            f"got {len(expert_tensors)}"
        )

    stacked = []
    for eid in range(n_total_experts):
        if eid not in expert_tensors:
            raise ValueError(f"Missing expert {eid} for {proj_type}.{suffix}")
        stacked.append(expert_tensors[eid])

    return torch.stack(stacked, dim=0)


def shard_expert_tensor(
    stacked: torch.Tensor,
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    """Extract the expert shard for a given EP rank.

    Args:
        stacked: [n_total_experts, ...] full expert tensor.
        ep_size: Expert parallelism group size.
        ep_rank: This rank's index within the EP group.

    Returns:
        [n_local, ...] tensor for this EP rank.
    """
    n_total = stacked.shape[0]
    if n_total % ep_size != 0:
        raise ValueError(
            f"n_total_experts ({n_total}) not divisible by ep_size ({ep_size})"
        )
    n_local = n_total // ep_size
    start = ep_rank * n_local
    end = start + n_local
    return stacked[start:end].clone()


# ---------------------------------------------------------------------------
# Name resolution: map HF key to nmoe key
# ---------------------------------------------------------------------------

def resolve_hf_key_to_nmoe(
    hf_key: str,
    base_mapping: Dict[str, str],
    n_layers: int,
) -> Optional[str]:
    """Resolve an HF key (including NVFP4 suffixes) to an nmoe key.

    Strategy:
      1. Direct lookup: try hf_key in base_mapping (for BF16 keys like norms/gates).
      2. NVFP4 lookup: strip suffix, look up base, re-append suffix.

    Args:
        hf_key: Full HF tensor name.
        base_mapping: The HF-base -> nmoe-base mapping from build_hf_to_nmoe_base_mapping.
        n_layers: Max layer count (keys beyond this are silently skipped).

    Returns:
        nmoe key string, or None if unmappable.
    """
    # 1. Direct lookup (BF16 keys: norms, gates, embed, lm_head)
    if hf_key in base_mapping:
        return base_mapping[hf_key]

    # 2. NVFP4 triplet key: strip suffix, lookup base, reattach suffix
    base, suffix = strip_nvfp4_suffix(hf_key)
    if base is not None and base in base_mapping:
        nmoe_base = base_mapping[base]
        return f"{nmoe_base}.{suffix}"

    # 3. Out-of-range layer: silently skip
    layer_match = re.match(r"model\.layers\.(\d+)\.", hf_key)
    if layer_match and int(layer_match.group(1)) >= n_layers:
        return None  # Expected: skip silently

    return None  # Truly unmapped


# ---------------------------------------------------------------------------
# Main import pipeline
# ---------------------------------------------------------------------------

def import_nvfp4_to_nmoe(
    model_dir: str,
    output_dir: str,
    ep_size: int = 8,
    n_layers: int = REAP_N_LAYERS,
    n_dense_layers: int = REAP_N_DENSE_LAYERS,
    n_routed_experts: int = REAP_N_ROUTED_EXPERTS,
    n_shared_experts: int = REAP_N_SHARED_EXPERTS,
) -> None:
    """Import HuggingFace NVFP4 compressed-tensors model to nmoe checkpoint format.

    This is the main entry point. NVFP4 triplets are kept intact -- no
    decompression or re-quantization. Expert weights are sharded across EP ranks
    with all 3 triplet components going to the same rank.

    The checkpoint stores NVFP4 format metadata so training can load expert
    triplets directly into NVFP4 primary buffers (no expert dequant/re-quant).

    Args:
        model_dir: Path to HF NVFP4 model directory with safetensors.
        output_dir: Output directory for nmoe checkpoint (iteration_XXXXX).
        ep_size: Expert parallelism size (default 8).
        n_layers: Total transformer layers.
        n_dense_layers: Number of initial dense (non-MoE) layers.
        n_routed_experts: Number of routed experts per MoE layer.
        n_shared_experts: Number of shared experts.
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_local = n_routed_experts // ep_size
    logger.info(f"Importing NVFP4 HF model from {model_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"EP size: {ep_size} -> {n_local} experts per rank")
    logger.info("Expert weights: NVFP4 compressed_tensors (preserved intact)")

    # 1. Load weight map from index
    weight_map = load_safetensors_index(model_dir)
    shard_groups = group_weights_by_shard(weight_map)
    logger.info(
        f"Model has {len(weight_map)} weight tensors across "
        f"{len(shard_groups)} shards"
    )

    # 2. Classify all keys
    expert_keys: Set[str] = set()
    non_expert_keys: Set[str] = set()

    for key in weight_map:
        info = parse_expert_nvfp4_key(key)
        if info is not None:
            expert_keys.add(key)
        else:
            non_expert_keys.add(key)

    logger.info(f"Expert tensor keys: {len(expert_keys)}")
    logger.info(f"Non-expert tensor keys: {len(non_expert_keys)}")

    n_moe_layers = n_layers - n_dense_layers
    # Each expert has 3 projections x 3 NVFP4 suffixes = 9 keys
    expected_expert_keys = n_routed_experts * 3 * 3 * n_moe_layers
    if len(expert_keys) != expected_expert_keys:
        logger.warning(
            f"Expected {expected_expert_keys} expert keys "
            f"(= {n_routed_experts} experts x 3 proj x 3 suffixes x {n_moe_layers} layers), "
            f"got {len(expert_keys)}"
        )

    # 3. Build base name mapping
    base_mapping = build_hf_to_nmoe_base_mapping(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        n_shared_experts=n_shared_experts,
    )

    # 4. Load non-expert weights -> rd.pt (dense/router state)
    logger.info("Loading non-expert weights for rd.pt...")
    dense_state: Dict[str, torch.Tensor] = {}
    unmapped_keys: List[str] = []

    for shard_name in sorted(shard_groups.keys()):
        shard_keys = shard_groups[shard_name]
        non_expert_in_shard = [k for k in shard_keys if k in non_expert_keys]
        if not non_expert_in_shard:
            continue

        logger.info(f"  Loading {len(non_expert_in_shard)} non-expert keys from {shard_name}")
        tensors = load_shard_tensors(model_dir, shard_name, non_expert_in_shard)

        for hf_name, tensor in tensors.items():
            nmoe_name = resolve_hf_key_to_nmoe(hf_name, base_mapping, n_layers)
            if nmoe_name is not None:
                dense_state[nmoe_name] = tensor
            else:
                # Check if out-of-range layer (skip silently)
                layer_match = re.match(r"model\.layers\.(\d+)\.", hf_name)
                if layer_match and int(layer_match.group(1)) >= n_layers:
                    pass
                else:
                    unmapped_keys.append(hf_name)

    if unmapped_keys:
        logger.warning(
            f"{len(unmapped_keys)} HF keys not mapped to nmoe names:"
        )
        for k in sorted(unmapped_keys)[:20]:
            logger.warning(f"  {k}")

    # Count BF16 vs NVFP4 in dense state
    bf16_count = sum(1 for k in dense_state if not any(k.endswith(s) for s in
                     ("weight_packed", "weight_scale", "weight_global_scale")))
    nvfp4_count = sum(1 for k in dense_state if k.endswith("weight_packed"))
    logger.info(
        f"Dense state dict: {len(dense_state)} keys "
        f"({bf16_count} BF16, {nvfp4_count} NVFP4 triplet groups)"
    )

    # 5. Build rd.pt
    rd_state = {
        "checkpoint_version": CHECKPOINT_FORMAT_VERSION,
        "step": 0,
        "tokens": 0,
        "model_dense": dense_state,
        "config_fingerprint": "",  # Empty: fingerprint bypass for imported checkpoints
        "nvfp4_format": "compressed_tensors",  # Signal to load path
        "nvfp4_group_size": 16,  # compressed_tensors group_size
        "run_info": {
            "git_sha": "import_from_hf_nvfp4",
            "preset": "dsv3_reap",
            "dtype": "nvfp4",
            "E": n_routed_experts,
            "K": 8,  # n_activated_experts
            "H": REAP_DIM,
            "L": n_layers,
            "world": ep_size,
            "dataset_version": None,
            "tokenizer_id": "deepseek_v3.2",
        },
        "imported_from": str(model_dir),
    }

    rd_path = output_dir / "rd.pt"
    logger.info(f"Saving rd.pt ({len(dense_state)} keys)...")
    torch.save(rd_state, str(rd_path))
    rd_size_gb = rd_path.stat().st_size / (1024 ** 3)
    logger.info(f"Saved rd.pt: {rd_size_gb:.2f} GB")

    # Free dense state
    del dense_state, rd_state

    logger.info("Processing expert NVFP4 triplets layer-by-layer...")

    # Group expert keys by (layer_id, proj_type, suffix)
    # Structure: expert_index[layer_id][proj_type][suffix][expert_id] = hf_key
    expert_index: Dict[int, Dict[str, Dict[str, Dict[int, str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for key in expert_keys:
        info = parse_expert_nvfp4_key(key)
        if info and info["layer_id"] < n_layers:
            expert_index[info["layer_id"]][info["proj_type"]][info["suffix"]][
                info["expert_id"]
            ] = key

    logger.info(
        f"Processing {len(expert_index)} MoE layers "
        f"(layers {n_dense_layers}-{n_layers - 1})"
    )

    # Initialize per-rank expert state dicts
    rank_states: Dict[int, Dict[str, torch.Tensor]] = {
        rank: {} for rank in range(ep_size)
    }

    # Shard cache to avoid reloading
    loaded_shard_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    for layer_id in sorted(expert_index.keys()):
        layer_t0 = time.time()
        logger.info(
            f"  Layer {layer_id}/{n_layers - 1} "
            f"(MoE layer {layer_id - n_dense_layers + 1}/{n_moe_layers})"
        )

        for proj_type in ("W1", "W3", "W2"):
            for suffix in ("weight_packed", "weight_scale", "weight_global_scale"):
                eid_to_key = expert_index[layer_id][proj_type][suffix]

                if len(eid_to_key) != n_routed_experts:
                    raise ValueError(
                        f"Layer {layer_id} {proj_type}.{suffix}: expected "
                        f"{n_routed_experts} experts, found {len(eid_to_key)}"
                    )

                # Collect tensors for all experts
                expert_tensors: Dict[int, torch.Tensor] = {}

                # Determine needed shards
                needed_shards: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
                for eid, hf_key in eid_to_key.items():
                    shard = weight_map[hf_key]
                    needed_shards[shard].append((eid, hf_key))

                for shard_name, eid_keys in needed_shards.items():
                    if shard_name not in loaded_shard_cache:
                        # Load all expert keys from this shard (not just current suffix)
                        keys_in_shard = [
                            k for k in weight_map
                            if weight_map[k] == shard_name and k in expert_keys
                        ]
                        loaded_shard_cache[shard_name] = load_shard_tensors(
                            model_dir, shard_name, keys_in_shard
                        )

                    shard_tensors = loaded_shard_cache[shard_name]
                    for eid, hf_key in eid_keys:
                        expert_tensors[eid] = shard_tensors[hf_key]

                # Stack into [n_total_experts, ...]
                stacked = stack_expert_nvfp4_triplets(
                    expert_tensors, n_routed_experts, proj_type, suffix,
                )
                del expert_tensors

                # Shard and store for each EP rank
                nmoe_key = f"blocks.{layer_id}.ffn.{proj_type}.{suffix}"

                for rank in range(ep_size):
                    rank_states[rank][nmoe_key] = shard_expert_tensor(
                        stacked, ep_size, rank
                    )

                del stacked

        # Periodically trim shard cache to limit memory
        if len(loaded_shard_cache) > 6:
            oldest_keys = sorted(loaded_shard_cache.keys())[:-3]
            for k in oldest_keys:
                del loaded_shard_cache[k]

        layer_dt = time.time() - layer_t0
        logger.info(f"    Layer {layer_id} done in {layer_dt:.1f}s")

    del loaded_shard_cache

    # 7. Save dp_rank_*.pt for each EP rank
    logger.info(f"Saving {ep_size} EP rank files...")

    for rank in range(ep_size):
        dp_state = {
            "checkpoint_version": CHECKPOINT_FORMAT_VERSION,
            "step": 0,
            "model_expert": rank_states[rank],
            "optimizer": {},  # No optimizer state for imported checkpoint
            "loader": None,
            "rng": None,
            "config_fingerprint": "",
            "nvfp4_format": "compressed_tensors",
            "nvfp4_group_size": 16,
            "ep_shard_info": {
                "ep_size": ep_size,
                "ep_rank": rank,
                "n_total_experts": n_routed_experts,
                "n_local_experts": n_local,
                "expert_start": rank * n_local,
                "expert_end": (rank + 1) * n_local,
            },
        }

        dp_path = output_dir / f"dp_rank_{rank:03d}.pt"
        logger.info(
            f"  Saving dp_rank_{rank:03d}.pt "
            f"({len(rank_states[rank])} keys)..."
        )
        torch.save(dp_state, str(dp_path))
        dp_size_gb = dp_path.stat().st_size / (1024 ** 3)
        logger.info(f"  Saved dp_rank_{rank:03d}.pt: {dp_size_gb:.2f} GB")

        del rank_states[rank]

    total_time = time.time() - t0
    logger.info(f"Import complete in {total_time:.1f}s")
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"  rd.pt: dense/router weights "
        f"(embedding, attn NVFP4 triplets, norms BF16, router BF16, shared experts NVFP4)"
    )
    logger.info(
        f"  dp_rank_000..{ep_size - 1:03d}.pt: "
        f"expert NVFP4 triplet shards ({n_local} experts each)"
    )
    logger.info("  NVFP4 triplets preserved intact (no decompression)")


# ---------------------------------------------------------------------------
# Validation / dry-run
# ---------------------------------------------------------------------------

def validate_import(
    model_dir: str,
    n_layers: int = REAP_N_LAYERS,
    n_dense_layers: int = REAP_N_DENSE_LAYERS,
    n_routed_experts: int = REAP_N_ROUTED_EXPERTS,
    n_shared_experts: int = REAP_N_SHARED_EXPERTS,
) -> bool:
    """Validate NVFP4 HF model directory and mapping without loading weights.

    Checks:
    - All expected HF weight keys exist in the index
    - All map to valid nmoe names
    - Expert NVFP4 triplet counts match expected
    - BF16 keys (norms, gates, embed, lm_head) are present

    Returns:
        True if validation passes, False otherwise.
    """
    model_dir = Path(model_dir)
    weight_map = load_safetensors_index(model_dir)

    base_mapping = build_hf_to_nmoe_base_mapping(
        n_layers, n_dense_layers, n_shared_experts,
    )

    all_hf_keys = set(weight_map.keys())
    expert_keys: Set[str] = set()
    mapped_keys: Set[str] = set()
    unmapped_keys: Set[str] = set()

    for key in all_hf_keys:
        info = parse_expert_nvfp4_key(key)
        if info is not None:
            expert_keys.add(key)
        else:
            nmoe_name = resolve_hf_key_to_nmoe(key, base_mapping, n_layers)
            if nmoe_name is not None:
                mapped_keys.add(key)
            else:
                # Check out-of-range layer
                layer_match = re.match(r"model\.layers\.(\d+)\.", key)
                if layer_match and int(layer_match.group(1)) >= n_layers:
                    pass  # Expected
                else:
                    unmapped_keys.add(key)

    # Categorize BF16 vs NVFP4 in non-expert keys
    bf16_mapped = {k for k in mapped_keys if not is_nvfp4_key(k)}
    nvfp4_mapped = {k for k in mapped_keys if is_nvfp4_key(k)}

    logger.info(f"Total HF keys: {len(all_hf_keys)}")
    logger.info(f"Expert keys (NVFP4 triplets): {len(expert_keys)}")
    logger.info(f"Non-expert mapped: {len(mapped_keys)} ({len(bf16_mapped)} BF16, {len(nvfp4_mapped)} NVFP4)")
    logger.info(f"Unmapped: {len(unmapped_keys)}")

    ok = True

    if unmapped_keys:
        logger.warning(f"Unmapped keys ({len(unmapped_keys)}):")
        for k in sorted(unmapped_keys)[:20]:
            logger.warning(f"  {k}")
        ok = False

    # Verify expert triplet counts per layer
    n_moe_layers = n_layers - n_dense_layers
    for layer_id in range(n_dense_layers, n_layers):
        for proj_name, proj_hf in [("gate_proj", "W1"), ("up_proj", "W3"), ("down_proj", "W2")]:
            for suffix in ("weight_packed", "weight_scale", "weight_global_scale"):
                pattern = f"model.layers.{layer_id}.mlp.experts."
                count = sum(
                    1 for k in expert_keys
                    if k.startswith(pattern) and proj_name in k and k.endswith(suffix)
                )
                if count != n_routed_experts:
                    logger.error(
                        f"Layer {layer_id} {proj_name}.{suffix}: "
                        f"expected {n_routed_experts} expert keys, found {count}"
                    )
                    ok = False

    # Verify BF16 essentials
    essential_bf16 = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for key in essential_bf16:
        if key not in all_hf_keys:
            logger.error(f"Missing essential BF16 key: {key}")
            ok = False
        elif key in mapped_keys:
            logger.info(f"  Essential BF16: {key} -> OK")

    if ok:
        logger.info("Validation PASSED.")
    else:
        logger.error("Validation FAILED -- see errors above.")

    return ok


# ---------------------------------------------------------------------------
# Plan-only mode: show sharding plan without loading tensors
# ---------------------------------------------------------------------------

def plan_sharding(
    model_dir: str,
    ep_size: int = 8,
    n_layers: int = REAP_N_LAYERS,
    n_dense_layers: int = REAP_N_DENSE_LAYERS,
    n_routed_experts: int = REAP_N_ROUTED_EXPERTS,
    n_shared_experts: int = REAP_N_SHARED_EXPERTS,
) -> None:
    """Plan and display the sharding strategy without loading any tensors.

    This is a structural test that:
    - Reads the safetensors index
    - Classifies all keys (expert vs non-expert, BF16 vs NVFP4)
    - Shows the mapping plan
    - Computes expected per-rank sizes
    """
    model_dir = Path(model_dir)
    weight_map = load_safetensors_index(model_dir)

    n_local = n_routed_experts // ep_size
    n_moe_layers = n_layers - n_dense_layers

    # Classify keys
    expert_keys: Set[str] = set()
    non_expert_keys: Set[str] = set()
    for key in weight_map:
        info = parse_expert_nvfp4_key(key)
        if info is not None:
            expert_keys.add(key)
        else:
            non_expert_keys.add(key)

    # Further classify non-expert
    bf16_non_expert = {k for k in non_expert_keys if not is_nvfp4_key(k)}
    nvfp4_non_expert = {k for k in non_expert_keys if is_nvfp4_key(k)}

    # Resolve names
    base_mapping = build_hf_to_nmoe_base_mapping(n_layers, n_dense_layers, n_shared_experts)
    mapped_count = 0
    unmapped_count = 0
    for key in non_expert_keys:
        nmoe_name = resolve_hf_key_to_nmoe(key, base_mapping, n_layers)
        if nmoe_name is not None:
            mapped_count += 1
        else:
            layer_match = re.match(r"model\.layers\.(\d+)\.", key)
            if layer_match and int(layer_match.group(1)) >= n_layers:
                pass  # Skip
            else:
                unmapped_count += 1

    # Expert breakdown by suffix
    expert_packed = sum(1 for k in expert_keys if k.endswith("weight_packed"))
    expert_scale = sum(1 for k in expert_keys if k.endswith("weight_scale"))
    expert_gscale = sum(1 for k in expert_keys if k.endswith("weight_global_scale"))

    print("=" * 72)
    print("NVFP4 Import Sharding Plan")
    print("=" * 72)
    print(f"Source model:  {model_dir}")
    print(f"Architecture:  DeepSeek-V3.2-REAP-345B-NVFP4A16-v2")
    print(f"Total layers:  {n_layers} ({n_dense_layers} dense + {n_moe_layers} MoE)")
    print(f"Experts:       {n_routed_experts} routed + {n_shared_experts} shared")
    print(f"EP size:       {ep_size} -> {n_local} experts per rank")
    print()
    print(f"Total tensor keys:  {len(weight_map)}")
    print(f"  Expert keys:      {len(expert_keys)}")
    print(f"    weight_packed:  {expert_packed}")
    print(f"    weight_scale:   {expert_scale}")
    print(f"    weight_global_scale: {expert_gscale}")
    print(f"  Non-expert keys:  {len(non_expert_keys)}")
    print(f"    BF16:           {len(bf16_non_expert)}")
    print(f"    NVFP4 triplets: {len(nvfp4_non_expert)}")
    print(f"  Mapped:           {mapped_count}")
    print(f"  Unmapped:         {unmapped_count}")
    print()

    per_expert_mb = 3 * (7.0 + 0.9 + 0.004)  # packed + scale + global_scale
    per_rank_expert_mb = n_local * n_moe_layers * per_expert_mb
    per_rank_expert_gb = per_rank_expert_mb / 1024

    print(f"Estimated per-rank expert shard: ~{per_rank_expert_gb:.1f} GB "
          f"({n_local} experts x {n_moe_layers} layers)")
    print(f"Estimated total expert data:     ~{per_rank_expert_gb * ep_size:.1f} GB")
    print()

    # Verify expert key integrity
    errors = 0
    for layer_id in range(n_dense_layers, n_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_packed", "weight_scale", "weight_global_scale"):
                count = sum(
                    1 for k in expert_keys
                    if f"layers.{layer_id}.mlp.experts." in k
                    and proj in k and k.endswith(suffix)
                )
                if count != n_routed_experts:
                    print(
                        f"  ERROR: Layer {layer_id} {proj}.{suffix}: "
                        f"expected {n_routed_experts}, got {count}"
                    )
                    errors += 1

    if errors == 0:
        print("Expert key integrity: PASSED (all layers have complete triplets)")
    else:
        print(f"Expert key integrity: FAILED ({errors} mismatches)")

    # Show shard distribution
    shards_with_experts = set()
    for k in expert_keys:
        shards_with_experts.add(weight_map[k])
    shards_non_expert_only = set(weight_map[k] for k in non_expert_keys) - shards_with_experts

    print()
    print(f"Safetensors shards: {len(set(weight_map.values()))}")
    print(f"  With expert data: {len(shards_with_experts)}")
    print(f"  Non-expert only:  {len(shards_non_expert_only)}")
    print()
    print(f"Checkpoint output structure:")
    print(f"  rd.pt:           dense/router/shared/attention weights")
    print(f"  dp_rank_000..{ep_size-1:03d}.pt:  {n_local} expert shards each (NVFP4 triplets)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Import HuggingFace NVFP4 compressed-tensors model to nmoe checkpoint format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to HuggingFace NVFP4 model directory with safetensors",
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Output directory for nmoe checkpoint (e.g., /data/ckpt/iteration_00000). "
             "Required unless --validate-only or --plan-only is used.",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=8,
        help="Expert parallelism size (default: 8)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=REAP_N_LAYERS,
        help=f"Total transformer layers (default: {REAP_N_LAYERS})",
    )
    parser.add_argument(
        "--n-dense-layers",
        type=int,
        default=REAP_N_DENSE_LAYERS,
        help=f"Number of initial dense layers (default: {REAP_N_DENSE_LAYERS})",
    )
    parser.add_argument(
        "--n-routed-experts",
        type=int,
        default=REAP_N_ROUTED_EXPERTS,
        help=f"Number of routed experts (default: {REAP_N_ROUTED_EXPERTS})",
    )
    parser.add_argument(
        "--n-shared-experts",
        type=int,
        default=REAP_N_SHARED_EXPERTS,
        help=f"Number of shared experts (default: {REAP_N_SHARED_EXPERTS})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate mapping without loading weights",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Show sharding plan without loading tensors (structural test)",
    )

    args = parser.parse_args()

    if not args.validate_only and not args.plan_only and args.output is None:
        parser.error("--output is required unless --validate-only or --plan-only is used")

    if args.validate_only:
        ok = validate_import(
            model_dir=args.model,
            n_layers=args.n_layers,
            n_dense_layers=args.n_dense_layers,
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=args.n_shared_experts,
        )
        sys.exit(0 if ok else 1)
    elif args.plan_only:
        plan_sharding(
            model_dir=args.model,
            ep_size=args.ep_size,
            n_layers=args.n_layers,
            n_dense_layers=args.n_dense_layers,
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=args.n_shared_experts,
        )
    else:
        import_nvfp4_to_nmoe(
            model_dir=args.model,
            output_dir=args.output,
            ep_size=args.ep_size,
            n_layers=args.n_layers,
            n_dense_layers=args.n_dense_layers,
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=args.n_shared_experts,
        )


if __name__ == "__main__":
    main()
