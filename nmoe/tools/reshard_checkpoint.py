#!/usr/bin/env python3
"""Checkpoint resharding utility for nmoe.

This tool converts checkpoints between different Expert Parallelism (EP)
configurations offline. For example, you can convert a checkpoint saved
with EP=4 to EP=8 for training on a larger cluster.

Usage:
    python -m nmoe.tools.reshard_checkpoint \\
        --input /path/to/checkpoint \\
        --output /path/to/output \\
        --step 10000 \\
        --source-ep 4 \\
        --target-ep 8

Example workflows:
    # Scale up: 4 GPUs -> 8 GPUs
    python -m nmoe.tools.reshard_checkpoint \\
        --input /checkpoints/run1 \\
        --output /checkpoints/run1_ep8 \\
        --step 10000 \\
        --target-ep 8

    # Scale down: 8 GPUs -> 2 GPUs
    python -m nmoe.tools.reshard_checkpoint \\
        --input /checkpoints/run2 \\
        --output /checkpoints/run2_ep2 \\
        --step 5000 \\
        --target-ep 2
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reshard nmoe checkpoints between EP configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input checkpoint directory (containing iter_XXXXXXX/)'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output checkpoint directory'
    )
    parser.add_argument(
        '--step', '-s',
        type=int,
        default=None,
        help='Step number to reshard (default: latest)'
    )
    parser.add_argument(
        '--source-ep',
        type=int,
        default=None,
        help='Source EP size (default: auto-detect from checkpoint)'
    )
    parser.add_argument(
        '--target-ep', '-t',
        type=int,
        required=True,
        help='Target EP size'
    )
    parser.add_argument(
        '--n-experts',
        type=int,
        default=None,
        help='Total number of experts (default: auto-detect from checkpoint)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate without writing output'
    )
    return parser.parse_args()


def find_latest_step(base_path: str) -> int:
    """Find the latest checkpointed step."""
    from nmoe.checkpoint import read_tracker
    step = read_tracker(base_path)
    if step <= 0:
        raise ValueError(f"No valid checkpoint found in {base_path}")
    return step


def detect_ep_size(it_dir: str) -> int:
    """Detect EP size from checkpoint directory."""
    # Count dp_rank_*.pt files
    dp_files = list(Path(it_dir).glob('dp_rank_*.pt'))
    if not dp_files:
        raise ValueError(f"No dp_rank_*.pt files found in {it_dir}")
    return len(dp_files)


def detect_n_experts(it_dir: str) -> int:
    """Detect total number of experts from checkpoint."""
    from nmoe.checkpoint import EPShardInfo, get_ep_shard_info_from_checkpoint

    # Load any dp_rank file to get EP info
    dp_files = sorted(Path(it_dir).glob('dp_rank_*.pt'))
    if not dp_files:
        raise ValueError(f"No dp_rank_*.pt files found in {it_dir}")

    ckpt = torch.load(dp_files[0], map_location='cpu', weights_only=False)
    ep_info = get_ep_shard_info_from_checkpoint(ckpt)

    if ep_info is not None:
        return ep_info.n_total_experts

    # Fallback: infer from first expert weight shape
    expert_sd = ckpt.get('model_expert', {})
    for key, tensor in expert_sd.items():
        if tensor.ndim >= 1:
            # Assume first dim is number of local experts
            n_local = tensor.shape[0]
            ep_size = len(dp_files)
            return n_local * ep_size

    raise ValueError("Cannot detect number of experts from checkpoint")


def load_and_gather_experts(
    it_dir: str,
    source_ep_size: int,
) -> Dict[str, torch.Tensor]:
    """Load all expert shards and gather into full tensors."""
    from nmoe.checkpoint import gather_all_expert_shards

    base_path = os.path.dirname(it_dir)
    step_str = os.path.basename(it_dir)
    step = int(step_str.replace('iter_', ''))

    return gather_all_expert_shards(base_path, step, source_ep_size)


def shard_experts_for_rank(
    full_experts: Dict[str, torch.Tensor],
    target_ep_size: int,
    target_rank: int,
    n_total_experts: int,
) -> Dict[str, torch.Tensor]:
    """Shard full expert weights for a specific EP rank."""
    n_local = n_total_experts // target_ep_size
    start = target_rank * n_local
    end = start + n_local

    sharded = {}
    for key, tensor in full_experts.items():
        if tensor.ndim >= 1:
            sharded[key] = tensor[start:end].contiguous()
        else:
            sharded[key] = tensor

    return sharded


def reshard_checkpoint(
    input_path: str,
    output_path: str,
    step: int,
    source_ep_size: int,
    target_ep_size: int,
    n_total_experts: int,
    dry_run: bool = False,
) -> None:
    """Reshard a checkpoint from source EP to target EP.

    Args:
        input_path: Input checkpoint base directory.
        output_path: Output checkpoint base directory.
        step: Step number to reshard.
        source_ep_size: Source EP size.
        target_ep_size: Target EP size.
        n_total_experts: Total number of experts.
        dry_run: If True, validate without writing.
    """
    from nmoe.checkpoint import (
        iteration_dir,
        EPShardInfo,
        CHECKPOINT_FORMAT_VERSION,
    )

    # Validate configuration
    if n_total_experts % target_ep_size != 0:
        raise ValueError(
            f"n_total_experts ({n_total_experts}) must be divisible by "
            f"target_ep_size ({target_ep_size})"
        )

    input_it_dir = iteration_dir(input_path, step)
    output_it_dir = iteration_dir(output_path, step)

    logger.info(f"Resharding checkpoint:")
    logger.info(f"  Input: {input_it_dir}")
    logger.info(f"  Output: {output_it_dir}")
    logger.info(f"  Source EP: {source_ep_size}")
    logger.info(f"  Target EP: {target_ep_size}")
    logger.info(f"  N experts: {n_total_experts}")

    if dry_run:
        logger.info("[DRY RUN] Validating only, no files will be written")

    # Step 1: Copy rd.pt (dense weights don't need resharding)
    input_rd = os.path.join(input_it_dir, 'rd.pt')
    output_rd = os.path.join(output_it_dir, 'rd.pt')

    logger.info(f"Loading dense weights from {input_rd}")
    rd_state = torch.load(input_rd, map_location='cpu', weights_only=False)

    # Update world size in run_info if present
    if 'run_info' in rd_state and rd_state['run_info'] is not None:
        rd_state['run_info']['world'] = target_ep_size
        logger.info(f"Updated run_info.world to {target_ep_size}")

    if not dry_run:
        os.makedirs(output_it_dir, exist_ok=True)
        torch.save(rd_state, output_rd)
        logger.info(f"Saved dense weights to {output_rd}")

    # Step 2: Gather all expert weights
    logger.info(f"Gathering expert weights from {source_ep_size} shards...")
    full_experts = load_and_gather_experts(input_it_dir, source_ep_size)
    logger.info(f"Gathered {len(full_experts)} expert weight tensors")

    for key, tensor in full_experts.items():
        logger.debug(f"  {key}: {tensor.shape}")

    # Step 3: Load template dp_rank for non-expert state
    template_dp = os.path.join(input_it_dir, 'dp_rank_000.pt')
    template_state = torch.load(template_dp, map_location='cpu', weights_only=False)

    # Step 4: Create new shards for target EP
    logger.info(f"Creating {target_ep_size} new shards...")

    for target_rank in range(target_ep_size):
        # Shard expert weights for this rank
        rank_experts = shard_experts_for_rank(
            full_experts, target_ep_size, target_rank, n_total_experts
        )

        # Create new EP shard info
        ep_info = EPShardInfo.create(
            ep_size=target_ep_size,
            ep_rank=target_rank,
            n_total_experts=n_total_experts,
        )

        # Build new state dict
        new_state = {
            'checkpoint_version': CHECKPOINT_FORMAT_VERSION,
            'step': template_state.get('step', step),
            'model_expert': rank_experts,
            'config_fingerprint': template_state.get('config_fingerprint', ''),
            'ep_shard_info': ep_info.to_dict(),
            # RNG state is per-rank, can't be meaningfully resharded
            'rng': None,
            # Optimizer state can't be resharded
            'optimizer': {},
            # Loader state is per-rank, keep as None
            'loader': None,
        }

        # Preserve zero2 if present (but note it won't be valid)
        if 'zero2' in template_state:
            logger.warning(f"Rank {target_rank}: ZeRO-2 state not resharded")

        output_dp = os.path.join(output_it_dir, f'dp_rank_{target_rank:03d}.pt')

        if not dry_run:
            torch.save(new_state, output_dp)
            logger.info(f"Saved rank {target_rank} shard: {output_dp}")
        else:
            expert_shapes = {k: v.shape for k, v in rank_experts.items()}
            logger.info(f"[DRY RUN] Would save rank {target_rank}: {expert_shapes}")

    # Step 5: Create manifest
    if not dry_run:
        files = ['rd.pt'] + [f'dp_rank_{i:03d}.pt' for i in range(target_ep_size)]
        bytes_total = sum(
            os.path.getsize(os.path.join(output_it_dir, f))
            for f in files if os.path.exists(os.path.join(output_it_dir, f))
        )

        manifest = {
            'step': step,
            'world': target_ep_size,
            'dp_count': target_ep_size,
            'bytes_total': bytes_total,
            'resharded_from_ep': source_ep_size,
            'files': files,
        }

        manifest_path = os.path.join(output_it_dir, 'manifest.json')
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved manifest to {manifest_path}")

        # Step 6: Write tracker
        from nmoe.checkpoint import write_tracker
        write_tracker(output_path, step)
        logger.info(f"Updated tracker to step {step}")

    logger.info("Resharding complete!")
    if not dry_run:
        logger.info(f"Output checkpoint ready at: {output_path}")
        logger.warning(
            "NOTE: Optimizer and RNG states were not resharded. "
            "First training step after resume will have fresh optimizer."
        )


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find step if not specified
    step = args.step
    if step is None:
        step = find_latest_step(args.input)
        logger.info(f"Using latest step: {step}")

    # Build iteration directory path
    from nmoe.checkpoint import iteration_dir
    input_it_dir = iteration_dir(args.input, step)

    if not os.path.exists(input_it_dir):
        logger.error(f"Checkpoint directory not found: {input_it_dir}")
        sys.exit(1)

    # Detect source EP if not specified
    source_ep = args.source_ep
    if source_ep is None:
        source_ep = detect_ep_size(input_it_dir)
        logger.info(f"Detected source EP size: {source_ep}")

    # Detect n_experts if not specified
    n_experts = args.n_experts
    if n_experts is None:
        n_experts = detect_n_experts(input_it_dir)
        logger.info(f"Detected total experts: {n_experts}")

    # Validate target EP
    if n_experts % args.target_ep != 0:
        logger.error(
            f"Total experts ({n_experts}) not divisible by "
            f"target EP ({args.target_ep})"
        )
        sys.exit(1)

    # Check if resharding is even needed
    if source_ep == args.target_ep:
        logger.warning(
            f"Source EP ({source_ep}) equals target EP ({args.target_ep}). "
            "No resharding needed."
        )
        if not args.dry_run:
            logger.info("Consider using 'cp -r' instead.")
        sys.exit(0)

    # Reshard
    try:
        reshard_checkpoint(
            input_path=args.input,
            output_path=args.output,
            step=step,
            source_ep_size=source_ep,
            target_ep_size=args.target_ep,
            n_total_experts=n_experts,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error(f"Resharding failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
