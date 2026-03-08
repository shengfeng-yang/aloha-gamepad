"""View LeRobot dataset using the LeRobotDataset API.

Usage:
    python view_lerobot.py <dataset_dir>
    python view_lerobot.py <dataset_dir> --episode 0
    python view_lerobot.py <dataset_dir> --episode 0 --camera cam_left_wrist
    python view_lerobot.py <dataset_dir> --summary

Example:
    python view_lerobot.py ../recorded_episodes/pick_book
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_dataset(dataset_dir: str) -> LeRobotDataset:
    ds_path = Path(dataset_dir).resolve()
    return LeRobotDataset(repo_id="local/dataset", root=str(ds_path))


def print_summary(ds: LeRobotDataset):
    info = ds.meta.info
    print("=" * 60)
    print(f"LeRobot Dataset: {ds.root}")
    print("=" * 60)
    print(f"  Version:    {info['codebase_version']}")
    print(f"  Robot:      {info.get('robot_type', 'unknown')}")
    print(f"  FPS:        {info['fps']}")
    print(f"  Episodes:   {info['total_episodes']}")
    print(f"  Frames:     {info['total_frames']}")

    print(f"\n  Features:")
    for name, ft in ds.meta.features.items():
        dtype = ft["dtype"]
        shape = ft["shape"]
        extra = ""
        if dtype == "video":
            vi = ft.get("info", {})
            extra = f"  ({vi.get('codec', '?')}, {vi.get('fps', '?')} fps)"
        print(f"    {name:45s} {dtype:10s} {str(shape):15s}{extra}")

    print(f"\n  Episodes:")
    for ep_idx in range(info["total_episodes"]):
        ep = ds.meta.episodes[ep_idx]
        length = ep["length"]
        print(f"    ep {ep_idx:3d}: {length:5d} frames ({length / info['fps']:.1f}s)")

    # Print sample data range
    print(f"\n  Sample data (frame 0):")
    s = ds[0]
    for key in ["observation.state", "action", "action.base"]:
        if key in s:
            v = s[key]
            print(f"    {key}: min={v.min().item():.3f} max={v.max().item():.3f}")


def _decode_frame(sample, vid_key):
    """Decode a video frame tensor to (H, W, C) uint8 numpy array."""
    img = sample[vid_key]
    if img.dim() == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    return (img.clamp(0, 1) * 255).to(torch.uint8).numpy()


def _get_motor_names(info):
    state_ft = info["features"].get("observation.state", {})
    names_info = state_ft.get("names", {})
    if isinstance(names_info, dict):
        return names_info.get("motors", [f"j{i}" for i in range(14)])
    return [f"j{i}" for i in range(14)]


def _get_base_names(info):
    base_ft = info["features"].get("action.base", {})
    names_info = base_ft.get("names", {})
    if isinstance(names_info, dict):
        return names_info.get("motors", ["linear_vel", "angular_vel"])
    return ["linear_vel", "angular_vel"]


def _collect_episode_data(ds, episode_index, camera_keys):
    """Read all frames for an episode and return arrays + thumbnail images."""
    info = ds.meta.info
    fps = info["fps"]
    ep = ds.meta.episodes[episode_index]
    ep_length = ep["length"]
    from_idx = ep["dataset_from_index"]

    thumb_indices = set(np.linspace(0, ep_length - 1, min(5, ep_length), dtype=int))

    states, actions, base_actions, timestamps = [], [], [], []
    # cam_key -> list of (frame_idx, image_array)
    cam_thumbs = {k: [] for k in camera_keys}

    for i in range(ep_length):
        sample = ds[from_idx + i]
        states.append(sample["observation.state"].numpy())
        actions.append(sample["action"].numpy())
        if "action.base" in sample:
            base_actions.append(sample["action.base"].numpy())
        timestamps.append(sample["timestamp"].item())

        if i in thumb_indices:
            for k in camera_keys:
                cam_thumbs[k].append((i, _decode_frame(sample, k)))

    return (
        np.array(states),
        np.array(actions),
        np.array(base_actions) if base_actions else None,
        np.array(timestamps),
        cam_thumbs,
    )


def _add_joint_and_base_rows(fig, n_rows, row, timestamps, states, base_actions, info):
    """Add left-arm, right-arm, and base-action subplot rows to *fig*."""
    motor_names = _get_motor_names(info)

    ax1 = fig.add_subplot(n_rows, 1, row + 1)
    for j in range(min(7, states.shape[1])):
        ax1.plot(timestamps, states[:, j], label=motor_names[j], linewidth=0.8)
    ax1.set_ylabel("Left Arm State")
    ax1.legend(fontsize=7, ncol=4, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(n_rows, 1, row + 2, sharex=ax1)
    for j in range(7, min(14, states.shape[1])):
        ax2.plot(timestamps, states[:, j], label=motor_names[j], linewidth=0.8)
    ax2.set_ylabel("Right Arm State")
    ax2.legend(fontsize=7, ncol=4, loc="upper right")
    ax2.grid(True, alpha=0.3)

    if base_actions is not None:
        base_names = _get_base_names(info)
        ax3 = fig.add_subplot(n_rows, 1, row + 3, sharex=ax1)
        for j in range(base_actions.shape[1]):
            ax3.plot(timestamps, base_actions[:, j], label=base_names[j], linewidth=1.0)
        ax3.set_ylabel("Base Action")
        ax3.set_xlabel("Time (s)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)


def view_episode(ds: LeRobotDataset, episode_index: int, camera: str, output_dir: Path):
    """Plot one camera row + joint/base plots."""
    info = ds.meta.info
    fps = info["fps"]
    if episode_index >= info["total_episodes"]:
        sys.exit(f"Episode {episode_index} out of range (0-{info['total_episodes'] - 1})")

    ep = ds.meta.episodes[episode_index]
    ep_length = ep["length"]
    from_idx = ep["dataset_from_index"]
    print(f"Episode {episode_index}: {ep_length} frames ({ep_length / fps:.1f}s)")
    print(f"  Global index range: [{from_idx}, {ep['dataset_to_index']})")

    vid_key = f"observation.images.{camera}"
    has_video = vid_key in ds.meta.video_keys
    cam_keys = [vid_key] if has_video else []

    states, actions, base_actions, timestamps, cam_thumbs = \
        _collect_episode_data(ds, episode_index, cam_keys)

    # Layout: 1 image row + 2 joint rows + optional base row
    n_data_rows = 3 if base_actions is not None else 2
    n_rows = n_data_rows + (1 if has_video else 0)

    fig = plt.figure(figsize=(18, 3.5 * n_rows))
    fig.suptitle(
        f"Episode {episode_index} -- {ep_length} frames @ {fps} fps -- {camera}",
        fontsize=14,
    )

    row = 0
    if has_video and cam_thumbs.get(vid_key):
        thumbs = cam_thumbs[vid_key]
        n_thumbs = len(thumbs)
        for i, (fi, img) in enumerate(thumbs):
            ax = fig.add_subplot(n_rows, n_thumbs, i + 1)
            ax.imshow(img)
            ax.set_title(f"t={fi / fps:.1f}s", fontsize=9)
            ax.axis("off")
        row += 1

    _add_joint_and_base_rows(fig, n_rows, row, timestamps, states, base_actions, info)

    plt.tight_layout()
    out_path = output_dir / f"episode_{episode_index}_{camera}.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def view_episode_all_cameras(ds: LeRobotDataset, episode_index: int, output_dir: Path):
    """Plot three camera rows (one per camera) + joint/base plots."""
    info = ds.meta.info
    fps = info["fps"]
    if episode_index >= info["total_episodes"]:
        sys.exit(f"Episode {episode_index} out of range (0-{info['total_episodes'] - 1})")

    ep = ds.meta.episodes[episode_index]
    ep_length = ep["length"]
    from_idx = ep["dataset_from_index"]
    print(f"Episode {episode_index}: {ep_length} frames ({ep_length / fps:.1f}s)")
    print(f"  Global index range: [{from_idx}, {ep['dataset_to_index']})")

    # Discover all video camera keys
    all_vid_keys = sorted(ds.meta.video_keys)
    # Derive short display names (e.g. "cam_high" from "observation.images.cam_high")
    cam_labels = {k: k.replace("observation.images.", "") for k in all_vid_keys}

    print(f"  Cameras: {', '.join(cam_labels.values())}")

    states, actions, base_actions, timestamps, cam_thumbs = \
        _collect_episode_data(ds, episode_index, all_vid_keys)

    n_cam_rows = len(all_vid_keys)
    n_data_rows = 3 if base_actions is not None else 2
    n_rows = n_cam_rows + n_data_rows
    n_thumbs = len(next(iter(cam_thumbs.values()), []))

    fig = plt.figure(figsize=(18, 3.0 * n_rows))
    fig.suptitle(
        f"Episode {episode_index} -- {ep_length} frames @ {fps} fps -- all cameras",
        fontsize=14,
    )

    # Image rows: one row per camera
    for cam_row, vid_key in enumerate(all_vid_keys):
        thumbs = cam_thumbs.get(vid_key, [])
        for i, (fi, img) in enumerate(thumbs):
            ax = fig.add_subplot(n_rows, n_thumbs, cam_row * n_thumbs + i + 1)
            ax.imshow(img)
            if cam_row == 0:
                ax.set_title(f"t={fi / fps:.1f}s", fontsize=9)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(cam_labels[vid_key], fontsize=9, rotation=0,
                              labelpad=60, va="center")

    # Joint / base rows
    _add_joint_and_base_rows(fig, n_rows, n_cam_rows, timestamps, states, base_actions, info)

    plt.tight_layout()
    out_path = output_dir / f"episode_{episode_index}_all_cameras.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View LeRobot dataset")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory (containing meta/, data/, videos/)")
    parser.add_argument("--episode", type=int, default=0, help="Episode index (default: 0)")
    parser.add_argument("--camera", type=str, default=None,
                        help="Show only this camera (e.g. cam_high, cam_left_wrist)")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: same as dataset_dir)")
    args = parser.parse_args()

    ds = load_dataset(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.dataset_dir)

    if args.summary:
        print_summary(ds)
    else:
        print_summary(ds)
        print()
        if args.camera:
            view_episode(ds, args.episode, args.camera, output_dir)
        else:
            view_episode_all_cameras(ds, args.episode, output_dir)
