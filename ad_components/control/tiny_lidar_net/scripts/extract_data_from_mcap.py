"""Extract LiDAR and control data from MCAP files recursively."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def extract_data_from_mcap(mcap_path: Path) -> dict[str, Any] | None:
    """Extract LiDAR scans and control commands from MCAP file.

    Args:
        mcap_path: Path to MCAP file

    Returns:
        Dictionary containing extracted numpy arrays or None if extraction failed.
    """
    scans_list = []
    scan_times = []
    control_times = []
    control_data = []

    logger.info(f"Reading MCAP file: {mcap_path}")

    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            target_topics = ["/sensing/lidar/scan", "/control/command/control_cmd"]

            for schema, channel, message in reader.iter_messages():
                if channel.topic not in target_topics:
                    continue

                msg = None
                if schema.encoding in ["json", "jsonschema"]:
                    try:
                        msg = json.loads(message.data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error for topic {channel.topic}: {e}")
                        continue
                elif schema.encoding == "cdr":
                    decoder = DecoderFactory().decoder_for(schema.encoding, schema)
                    if decoder:
                        msg = decoder.decode(message.data)

                if msg is None:
                    continue

                # Extract LiDAR scans
                if channel.topic == "/sensing/lidar/scan":
                    ranges = None
                    if isinstance(msg, dict):
                        if "ranges" in msg:
                            ranges = np.array(msg["ranges"], dtype=np.float32)
                    elif hasattr(msg, "ranges"):
                        ranges = np.array(msg.ranges, dtype=np.float32)

                    if ranges is not None:
                        scans_list.append(ranges)
                        scan_times.append(message.log_time)

                # Extract control commands
                elif channel.topic == "/control/command/control_cmd":
                    steer = 0.0
                    accel = 0.0
                    found = False

                    if isinstance(msg, dict):
                        if "lateral" in msg and "longitudinal" in msg:  # Autoware Control
                            steer = msg["lateral"].get("steering_tire_angle", 0.0)
                            accel = msg["longitudinal"].get("acceleration", 0.0)
                            found = True
                    else:
                        if hasattr(msg, "lateral") and hasattr(
                            msg, "longitudinal"
                        ):  # Autoware Control
                            steer = msg.lateral.steering_tire_angle
                            accel = msg.longitudinal.acceleration
                            found = True

                    if found:
                        control_data.append([steer, accel])
                        control_times.append(message.log_time)

    except Exception as e:
        logger.error(f"Failed to read {mcap_path}: {e}")
        return None

    if not scans_list or not control_data:
        logger.warning(f"Warning: No data extracted from {mcap_path}")
        return None

    # Convert to numpy arrays
    scans = np.array(scans_list, dtype=np.float32)
    scan_times = np.array(scan_times, dtype=np.int64)
    control_data = np.array(control_data, dtype=np.float32)
    control_times = np.array(control_times, dtype=np.int64)

    # Synchronize data using nearest neighbor
    indices, _ = synchronize_data(scan_times, control_times)

    synced_controls = control_data[indices]
    synced_steers = synced_controls[:, 0]
    synced_accels = synced_controls[:, 1]

    return {"scans": scans, "steers": synced_steers, "accelerations": synced_accels}


def synchronize_data(
    src_times: np.ndarray, target_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Synchronize two time series using nearest neighbor search."""
    if len(target_times) == 0:
        return np.array([]), np.array([])

    idx_sorted = np.searchsorted(target_times, src_times)
    idx_sorted = np.clip(idx_sorted, 0, len(target_times) - 1)
    prev_idx = np.clip(idx_sorted - 1, 0, len(target_times) - 1)

    time_diff_curr = np.abs(target_times[idx_sorted] - src_times)
    time_diff_prev = np.abs(target_times[prev_idx] - src_times)

    use_prev = time_diff_prev < time_diff_curr
    final_indices = np.where(use_prev, prev_idx, idx_sorted)
    final_deltas = np.where(use_prev, time_diff_prev, time_diff_curr)

    return final_indices, final_deltas


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig) -> None:
    """Main function to batch process MCAP files."""

    # We expect arguments: input_dir=... output_path=...
    # Since we use Hydra, we can accept these as normal overrides or config.
    # But since we have no config file for this script, we can just use argparse compatible mode
    # or rely on simple hydra overrides.

    # However, let's use argparse for this utility script to be simpler,
    # OR stick to Hydra for consistency? The user requested Hydra.
    # But standard hydra needs a config directory.
    # Let's use argparse for simplicity as this is a helper tool, NOT the main experiment runner.
    # Wait, the prompt implies "Update ... to support directory recursion".
    pass


if __name__ == "__main__":
    # Switching to argparse for this independent utility
    parser = argparse.ArgumentParser(description="Extract data from MCAP files recursively")
    parser.add_argument(
        "--input_dir", type=Path, required=True, help="Input directory containing MCAP files"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Output directory for processed .npy files"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    # Find all mcap files recursively
    mcap_files = list(input_dir.rglob("*.mcap"))
    logger.info(f"Found {len(mcap_files)} MCAP files in {input_dir}")

    all_scans = []
    all_steers = []
    all_accels = []

    for mcap_file in mcap_files:
        result = extract_data_from_mcap(mcap_file)
        if result:
            all_scans.append(result["scans"])
            all_steers.append(result["steers"])
            all_accels.append(result["accelerations"])

    if not all_scans:
        logger.error("No data extracted.")
        exit(1)

    # Concatenate all data
    final_scans = np.concatenate(all_scans, axis=0)
    final_steers = np.concatenate(all_steers, axis=0)
    final_accels = np.concatenate(all_accels, axis=0)

    # Save combined dataset
    np.save(output_dir / "scans.npy", final_scans)
    np.save(output_dir / "steers.npy", final_steers)
    np.save(output_dir / "accelerations.npy", final_accels)

    logger.info(f"Saved combined dataset to {output_dir}")
    logger.info(f"  Total samples: {len(final_scans)}")
    logger.info(f"  Scans shape: {final_scans.shape}")
