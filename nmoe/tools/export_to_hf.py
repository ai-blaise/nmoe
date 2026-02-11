"""Export nmoe models to HuggingFace format.

This tool exports nmoe checkpoints to HuggingFace-compatible format that can
be loaded by transformers AutoModelForCausalLM or served by SGLang.

Usage:
    python -m nmoe.tools.export_to_hf \
        --checkpoint /path/to/nmoe/checkpoint \
        --output /path/to/hf/output \
        --config /path/to/config.toml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch

from nmoe.unified.config import NMoEModelConfig
from nmoe.tools.config_converter import (
    expand_expert_weights_to_hf,
)


def generate_config_json(
    config: NMoEModelConfig,
    output_path: Path,
) -> Dict[str, Any]:
    """Generate HuggingFace config.json from NMoEModelConfig.

    Args:
        config: Unified model config
        output_path: Directory to save config.json

    Returns:
        The generated config dict
    """
    hf_config = config.to_hf_config()

    # Add HF-required metadata
    hf_config["transformers_version"] = "4.40.0"
    hf_config["auto_map"] = {
        "AutoConfig": "configuration_nmoe.NMoEConfig",
        "AutoModelForCausalLM": "modeling_nmoe.NMoEForCausalLM",
    }

    # Write config.json
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)

    print(f"Wrote config.json to {config_path}")
    return hf_config


def generate_generation_config(
    config: NMoEModelConfig,
    output_path: Path,
) -> Dict[str, Any]:
    """Generate HuggingFace generation_config.json.

    Args:
        config: Unified model config
        output_path: Directory to save generation_config.json

    Returns:
        The generated config dict
    """
    gen_config = {
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "max_length": config.max_position_embeddings,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "transformers_version": "4.40.0",
    }

    # Remove None values
    gen_config = {k: v for k, v in gen_config.items() if v is not None}

    config_path = output_path / "generation_config.json"
    with open(config_path, "w") as f:
        json.dump(gen_config, f, indent=2)

    print(f"Wrote generation_config.json to {config_path}")
    return gen_config


def load_nmoe_checkpoint(
    checkpoint_path: Path,
) -> Dict[str, torch.Tensor]:
    """Load nmoe checkpoint from directory.

    nmoe checkpoints have structure:
        iteration_XXXXX/
            rd.pt  (router/dense params)
            dp_rank_0.pt, dp_rank_1.pt, ... (expert params per DP rank)

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Merged state dict with all weights
    """
    checkpoint_path = Path(checkpoint_path)

    # Find latest iteration
    if checkpoint_path.name.startswith("iteration_"):
        iter_dir = checkpoint_path
    else:
        iter_dirs = sorted(checkpoint_path.glob("iteration_*"))
        if not iter_dirs:
            raise ValueError(f"No iteration_* directories in {checkpoint_path}")
        iter_dir = iter_dirs[-1]
        print(f"Using latest checkpoint: {iter_dir.name}")

    state_dict = {}

    # Load router/dense params
    rd_path = iter_dir / "rd.pt"
    if rd_path.exists():
        rd_state = torch.load(rd_path, map_location="cpu")
        state_dict.update(rd_state)
        print(f"Loaded {len(rd_state)} keys from rd.pt")

    # Load expert params from all DP ranks
    dp_files = sorted(iter_dir.glob("dp_rank_*.pt"))
    for dp_file in dp_files:
        dp_state = torch.load(dp_file, map_location="cpu")
        # Expert weights need to be gathered across ranks
        # For now, assume single-rank export
        state_dict.update(dp_state)
        print(f"Loaded {len(dp_state)} keys from {dp_file.name}")

    return state_dict


def export_weights_to_hf(
    nmoe_state_dict: Dict[str, torch.Tensor],
    config: NMoEModelConfig,
    output_path: Path,
    shard_size_gb: float = 5.0,
) -> None:
    """Export nmoe weights to HuggingFace safetensors format.

    Args:
        nmoe_state_dict: nmoe checkpoint state dict
        config: Model configuration
        output_path: Directory to save weights
        shard_size_gb: Maximum shard size in GB (default 5GB)
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("safetensors required: pip install safetensors")

    # Expand expert weights to individual tensors
    hf_state_dict = expand_expert_weights_to_hf(
        nmoe_state_dict,
        n_layers=config.num_hidden_layers,
        n_dense_layers=config.first_k_dense_replace,
        n_experts=config.num_experts,
    )

    # Calculate total size and determine sharding
    total_bytes = sum(t.numel() * t.element_size() for t in hf_state_dict.values())
    total_gb = total_bytes / (1024 ** 3)
    print(f"Total model size: {total_gb:.2f} GB")

    if total_gb <= shard_size_gb:
        # Single file
        output_file = output_path / "model.safetensors"
        save_file(hf_state_dict, output_file)
        print(f"Saved weights to {output_file}")

        # Write index
        weight_map = {k: "model.safetensors" for k in hf_state_dict.keys()}
        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
    else:
        # Shard the model
        shard_bytes = int(shard_size_gb * 1024 ** 3)
        shards = []
        current_shard = {}
        current_size = 0
        shard_idx = 1

        for key, tensor in hf_state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()

            if current_size + tensor_size > shard_bytes and current_shard:
                shards.append((f"model-{shard_idx:05d}-of-XXXXX.safetensors", current_shard))
                shard_idx += 1
                current_shard = {}
                current_size = 0

            current_shard[key] = tensor
            current_size += tensor_size

        if current_shard:
            shards.append((f"model-{shard_idx:05d}-of-XXXXX.safetensors", current_shard))

        # Update shard names with total count
        n_shards = len(shards)
        weight_map = {}
        for i, (name, shard) in enumerate(shards):
            final_name = name.replace("XXXXX", f"{n_shards:05d}")
            output_file = output_path / final_name
            save_file(shard, output_file)
            print(f"Saved shard {i+1}/{n_shards} to {final_name}")

            for key in shard.keys():
                weight_map[key] = final_name

        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }

    # Write model index
    index_path = output_path / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote index to {index_path}")


def export_nmoe_to_hf(
    checkpoint_path: str,
    output_path: str,
    config: Optional[NMoEModelConfig] = None,
    shard_size_gb: float = 5.0,
) -> None:
    """Export nmoe checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to nmoe checkpoint
        output_path: Output directory for HF model
        config: Model config (if None, infer from checkpoint)
        shard_size_gb: Maximum shard size in GB
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = load_nmoe_checkpoint(checkpoint_path)

    if config is None:
        # Try to load config from checkpoint
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = NMoEModelConfig.from_dict(config_dict)
        else:
            raise ValueError("Config required: pass config= or have config.json in checkpoint")

    # Generate config files
    generate_config_json(config, output_path)
    generate_generation_config(config, output_path)

    # Export weights
    export_weights_to_hf(state_dict, config, output_path, shard_size_gb)

    print(f"\nExport complete! Model saved to {output_path}")
    print("Load with: AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)")


def main():
    parser = argparse.ArgumentParser(description="Export nmoe model to HuggingFace format")
    parser.add_argument("--checkpoint", required=True, help="Path to nmoe checkpoint")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--shard-size", type=float, default=5.0, help="Shard size in GB")

    args = parser.parse_args()

    export_nmoe_to_hf(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        shard_size_gb=args.shard_size,
    )


if __name__ == "__main__":
    main()
