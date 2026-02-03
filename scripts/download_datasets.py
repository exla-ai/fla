#!/usr/bin/env python3
"""Download datasets for VLA fine-tuning.

This script downloads robotics datasets in LeRobot format from HuggingFace.

Supported datasets:
- DROID: 92k episodes, 27M frames (Franka)
- LIBERO: 130+ tasks benchmark
- ALOHA: Bimanual manipulation (sim + real)
- Open X-Embodiment: 60+ datasets, 22 robot types

Usage:
    # Download all recommended datasets
    python scripts/download_datasets.py

    # Download specific dataset group
    python scripts/download_datasets.py --group libero
    python scripts/download_datasets.py --group aloha
    python scripts/download_datasets.py --group droid

    # Download specific dataset by name
    python scripts/download_datasets.py --dataset lerobot/libero_10

    # List available datasets
    python scripts/download_datasets.py --list
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dataset groups organized by use case
DATASETS = {
    # LIBERO Benchmark - Recommended for evaluation
    "libero": {
        "description": "LIBERO manipulation benchmark (130+ tasks)",
        "datasets": [
            "lerobot/libero",           # Full LIBERO dataset
            "lerobot/libero_10",        # 10-task subset
            "lerobot/libero_10_image",  # With images
        ],
    },
    # ALOHA - Bimanual manipulation
    "aloha": {
        "description": "ALOHA bimanual robot datasets",
        "datasets": [
            # Simulation
            "lerobot/aloha_sim_transfer_cube_human",
            "lerobot/aloha_sim_transfer_cube_scripted",
            "lerobot/aloha_sim_insertion_human",
            "lerobot/aloha_sim_insertion_scripted",
            # Real robot
            "lerobot/aloha_static_towel",
            "lerobot/aloha_static_coffee",
            "lerobot/aloha_static_cups_open",
            "lerobot/aloha_static_battery",
            "lerobot/aloha_static_candy",
            # Mobile ALOHA
            "lerobot/aloha_mobile_cabinet",
            "lerobot/aloha_mobile_chair",
            "lerobot/aloha_mobile_elevator",
            "lerobot/aloha_mobile_wash_pan",
        ],
    },
    # DROID - Large-scale Franka dataset
    "droid": {
        "description": "DROID dataset (92k episodes, 27M frames)",
        "datasets": [
            "lerobot/droid_100",     # 100-episode sample (recommended for testing)
            # "lerobot/droid_1.0.1", # Full dataset (1.7TB) - uncomment if needed
        ],
    },
    # Open X-Embodiment - Cross-embodiment datasets
    "oxe": {
        "description": "Open X-Embodiment cross-robot datasets",
        "datasets": [
            "lerobot/stanford_kuka_multimodal_dataset",
            "lerobot/taco_play",
            "lerobot/jaco_play",
            "lerobot/berkeley_autolab_ur5",
            "lerobot/berkeley_cable_routing",
            "lerobot/berkeley_fanuc_manipulation",
            "lerobot/columbia_cairlab_pusht_real",
            "lerobot/cmu_franka_exploration_dataset",
            "lerobot/ucsd_kitchen_dataset",
            "lerobot/nyu_franka_play_dataset",
            "lerobot/utokyo_xarm_bimanual",
        ],
    },
    # Quick start - Minimal datasets for testing
    "quickstart": {
        "description": "Minimal datasets for quick testing",
        "datasets": [
            "lerobot/aloha_sim_transfer_cube_human",
            "lerobot/aloha_sim_insertion_human",
            "lerobot/libero_10",
        ],
    },
    # Cross-embodiment training
    "cross_embodiment": {
        "description": "Datasets for cross-embodiment training",
        "datasets": [
            # ALOHA (14 DOF)
            "lerobot/aloha_sim_transfer_cube_human",
            "lerobot/aloha_sim_insertion_human",
            # Single-arm (7 DOF)
            "lerobot/libero_10",
            "lerobot/berkeley_autolab_ur5",
            "lerobot/jaco_play",
            # Bimanual
            "lerobot/utokyo_xarm_bimanual",
        ],
    },
}


def download_dataset(repo_id: str) -> bool:
    """Download a single LeRobot dataset."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        logger.error("lerobot not installed. Run: pip install lerobot")
        return False

    try:
        logger.info(f"Downloading {repo_id}...")
        ds = LeRobotDataset(repo_id)
        logger.info(f"  ✓ {repo_id}: {ds.num_episodes} episodes, {ds.num_frames} frames")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed to download {repo_id}: {e}")
        return False


def download_group(group_name: str) -> None:
    """Download all datasets in a group."""
    if group_name not in DATASETS:
        logger.error(f"Unknown group: {group_name}")
        logger.info(f"Available groups: {', '.join(DATASETS.keys())}")
        return

    group = DATASETS[group_name]
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading: {group['description']}")
    logger.info(f"{'='*60}\n")

    success = 0
    failed = 0
    for repo_id in group["datasets"]:
        if download_dataset(repo_id):
            success += 1
        else:
            failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Complete: {success} succeeded, {failed} failed")
    logger.info(f"{'='*60}")


def list_datasets() -> None:
    """List all available dataset groups."""
    print("\n" + "="*60)
    print("Available Dataset Groups")
    print("="*60 + "\n")

    for group_name, group in DATASETS.items():
        print(f"  {group_name}:")
        print(f"    {group['description']}")
        print(f"    Datasets: {len(group['datasets'])}")
        print()

    print("Usage:")
    print("  python scripts/download_datasets.py --group quickstart  # Minimal for testing")
    print("  python scripts/download_datasets.py --group libero      # LIBERO benchmark")
    print("  python scripts/download_datasets.py --group aloha       # ALOHA bimanual")
    print("  python scripts/download_datasets.py --group cross_embodiment  # Multi-robot")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download robotics datasets for VLA fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_datasets.py --group quickstart
  python scripts/download_datasets.py --group libero --group aloha
  python scripts/download_datasets.py --dataset lerobot/libero_10
  python scripts/download_datasets.py --list
        """
    )
    parser.add_argument("--group", type=str, action="append",
                        help="Dataset group to download (can specify multiple)")
    parser.add_argument("--dataset", type=str, action="append",
                        help="Specific dataset to download (can specify multiple)")
    parser.add_argument("--list", action="store_true",
                        help="List available dataset groups")
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets (large!)")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.all:
        for group_name in DATASETS:
            download_group(group_name)
        return

    if args.dataset:
        for repo_id in args.dataset:
            download_dataset(repo_id)

    if args.group:
        for group_name in args.group:
            download_group(group_name)

    if not args.dataset and not args.group:
        # Default: download quickstart datasets
        logger.info("No group specified, downloading quickstart datasets...")
        logger.info("Use --list to see all available groups")
        download_group("quickstart")

    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("""
1. Train on downloaded data:
   python scripts/train.py pi06_multi --exp-name my_experiment

2. For cross-embodiment training:
   python scripts/train.py pi06_cross_embodiment --exp-name v1

3. Evaluate your model:
   python scripts/evaluate_aloha_sim.py --checkpoint_dir ./checkpoints/...
    """)


if __name__ == "__main__":
    main()
