import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from experiment.engine.base import BaseEngine
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from omegaconf import DictConfig

# 既存の抽出ロジック（ad_components配下）を参照する代わりに、ここに統合していく方針
# 今回は構造のデモとして骨格を実装

logger = logging.getLogger(__name__)


class ExtractorEngine(BaseEngine):
    """データ抽出・変換・統計量計算エンジン"""

    def _run_impl(self, cfg: DictConfig) -> Any:
        input_dir = Path(cfg.input_dir)
        # output_dir is mandatory in strict config
        output_dir = Path(cfg.output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting data from {input_dir} to {output_dir}")

        # 1. MCAP files discovery
        mcap_files = list(input_dir.rglob("*.mcap"))
        if not mcap_files:
            logger.error(f"No MCAP files found in {input_dir}")
            return None

        # 2. Extract and sync data from all MCAPs
        all_scans = []
        all_steers = []
        all_accels = []

        for mcap_file in mcap_files:
            result = self._extract_from_single_mcap(mcap_file)
            if result:
                all_scans.append(result["scans"])
                all_steers.append(result["steers"])
                all_accels.append(result["accelerations"])

        if not all_scans:
            logger.error("No data could be extracted from any MCAP file.")
            return None

        scans = np.concatenate(all_scans, axis=0)
        steers = np.concatenate(all_steers, axis=0)
        accels = np.concatenate(all_accels, axis=0)

        # 3. Calculate statistics
        # Filter finite values for stats calculation to avoid NaN
        valid_scans = scans[np.isfinite(scans)]
        if len(valid_scans) == 0:
            logger.warning("No finite scan values found for statistics.")
            valid_scans = np.zeros(1)

        stats = {
            "scans": {
                "mean": float(np.mean(valid_scans)),
                "std": float(np.std(valid_scans)),
                # Use numpy min/max to handle potentially empty valid set if logic above fails, though caught by len check
                "min": float(np.min(valid_scans)),
                "max": float(np.max(valid_scans)),
            },
            "steers": {
                "mean": float(np.mean(steers)),
                "std": float(np.std(steers)),
            },
            "accels": {
                "mean": float(np.mean(accels)),
                "std": float(np.std(accels)),
            },
        }

        # 4. Save processed data and stats
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        np.save(output_dir / "scans.npy", scans)
        np.save(output_dir / "steers.npy", steers)
        np.save(output_dir / "accelerations.npy", accels)

        # 5. Save metadata (Lineage)
        self._save_metadata(output_dir, input_dir)

        logger.info(f"Successfully extracted {len(scans)} samples.")
        logger.info(f"Dataset saved to {output_dir}")

        # 6. DVC Automation
        if cfg.dvc and cfg.dvc.auto_add:
            self._run_dvc_commands(output_dir, cfg.dvc.auto_push)

        return output_dir

    def _save_metadata(self, output_dir: Path, input_dir: Path) -> None:
        """Save metadata for data lineage."""
        metadata = {
            "input_dir": str(input_dir),
            "created_at": str(np.datetime64("now")),
            # Try to get git commit hash
            "git_commit": self._get_git_commit(),
        }
        try:
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "unknown"

    def _run_dvc_commands(self, target_dir: Path, push: bool = False) -> None:
        """Run dvc add and optionally dvc push."""
        try:
            logger.info(f"Running: dvc add {target_dir}")
            subprocess.run(["dvc", "add", str(target_dir)], check=True)

            if push:
                logger.info("Running: dvc push")
                subprocess.run(["dvc", "push"], check=True)

        except subprocess.CalledProcessError as e:
            logger.error(f"DVC command failed: {e}")
        except FileNotFoundError:
            logger.error("dvc command not found. Please ensure DVC is installed.")

    def _extract_from_single_mcap(self, mcap_path: Path) -> dict[str, Any] | None:
        """Extract and sync data from one MCAP."""
        scans_list = []
        scan_times = []
        control_times = []
        control_data = []

        try:
            with open(mcap_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                target_topics = ["/perception/lidar/scan", "/control/command/control_cmd"]

                for schema, channel, message in reader.iter_messages():
                    if channel.topic not in target_topics:
                        continue

                    msg = None
                    # Try to decode based on message encoding
                    if channel.message_encoding == "json":
                        try:
                            msg = json.loads(message.data)
                        except Exception as e:
                            logger.debug(f"JSON decode error on {channel.topic}: {e}")
                    elif schema and schema.encoding == "cdr":
                        decoder = DecoderFactory().decoder_for(schema.encoding, schema)
                        if decoder:
                            msg = decoder.decode(message.data)

                    if msg is None:
                        logger.debug(
                            f"Skipping message on {channel.topic} (unsupported encoding or decode failed)"
                        )
                        continue

                    logger.debug(f"Successfully decoded message on topic {channel.topic}")

                    if channel.topic == "/perception/lidar/scan":
                        ranges = None
                        if isinstance(msg, dict) and "ranges" in msg:
                            ranges = np.array(msg["ranges"], dtype=np.float32)
                        elif hasattr(msg, "ranges"):
                            ranges = np.array(msg.ranges, dtype=np.float32)

                        if ranges is not None:
                            # Replace inf/nan with 0.0 or a safe value
                            # For statistics, we might want to filter them out, but for dataset consistency we keep shape.
                            # Standard practice: replace inf with max_range or 0.
                            ranges = np.nan_to_num(ranges, posinf=30.0, neginf=0.0)
                            scans_list.append(ranges)
                            scan_times.append(message.log_time)

                    elif channel.topic == "/control/command/control_cmd":
                        steer, accel, found = 0.0, 0.0, False
                        if isinstance(msg, dict):
                            if "drive" in msg:
                                steer, accel, found = (
                                    msg["drive"].get("steering_angle", 0.0),
                                    msg["drive"].get("acceleration", 0.0),
                                    True,
                                )
                            elif "lateral" in msg and "longitudinal" in msg:
                                steer, accel, found = (
                                    msg["lateral"].get("steering_tire_angle", 0.0),
                                    msg["longitudinal"].get("acceleration", 0.0),
                                    True,
                                )
                        else:
                            if hasattr(msg, "drive"):
                                steer, accel, found = (
                                    msg.drive.steering_angle,
                                    msg.drive.acceleration,
                                    True,
                                )
                            elif hasattr(msg, "lateral"):
                                steer, accel, found = (
                                    msg.lateral.steering_tire_angle,
                                    msg.longitudinal.acceleration,
                                    True,
                                )

                        if found:
                            control_data.append([steer, accel])
                            control_times.append(message.log_time)
        except Exception as e:
            logger.error(f"Error reading {mcap_path}: {e}")
            return None

        if not scans_list or not control_data:
            logger.warning(
                f"Insufficient data in {mcap_path}: scans={len(scans_list)}, controls={len(control_data)}"
            )
            return None

        # Sync
        s_times = np.array(scan_times, dtype=np.int64)
        c_times = np.array(control_times, dtype=np.int64)
        c_data = np.array(control_data, dtype=np.float32)

        idx = np.searchsorted(c_times, s_times)
        idx = np.clip(idx, 0, len(c_times) - 1)

        synced_controls = c_data[idx]
        return {
            "scans": np.array(scans_list, dtype=np.float32),
            "steers": synced_controls[:, 0],
            "accelerations": synced_controls[:, 1],
        }
