import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from omegaconf import DictConfig

from experiment.engine.base import BaseEngine

# 既存の抽出ロジック（ad_components配下）を参照する代わりに、ここに統合していく方針
# 今回は構造のデモとして骨格を実装

logger = logging.getLogger(__name__)


class ExtractorEngine(BaseEngine):
    """データ抽出・変換・統計量計算エンジン"""

    def _run_impl(self, cfg: DictConfig) -> Any:
        input_dir = Path(cfg.input_dir)
        # output_dir is mandatory in strict config
        output_dir = Path(cfg.output_dir)

        # Initialize topics configuration
        self.topics = {}
        if "experiment" in cfg:
             self.topics = cfg.experiment.get("topics", {})

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Config keys: {list(cfg.keys())}")
        if "experiment" in cfg:
            logger.info(f"Experiment keys: {list(cfg.experiment.keys())}")

        # Get exclude_failure_reasons config
        exclude_reasons = cfg.get("exclude_failure_reasons", None)
        # Fallback to experiment.exclude_failure_reasons if not at root
        if exclude_reasons is None and "experiment" in cfg:
            exclude_reasons = cfg.experiment.get("exclude_failure_reasons", None)

        # Backward compatibility: handle deprecated include_failed_episodes
        include_failed_legacy = cfg.get("include_failed_episodes", None)
        if include_failed_legacy is None and "experiment" in cfg:
            include_failed_legacy = cfg.experiment.get("include_failed_episodes", None)
        if include_failed_legacy is not None:
            logger.warning(
                "include_failed_episodes is deprecated. Use exclude_failure_reasons instead. "
                "Treating as exclude_failure_reasons=[] (include all failures)"
            )
            if include_failed_legacy and exclude_reasons is None:
                exclude_reasons = []  # include all failures

        # Log filtering behavior
        if exclude_reasons is None:
            logger.info("Excluding all failed episodes (exclude_failure_reasons=null)")
        elif len(exclude_reasons) == 0:
            logger.info("Including all failed episodes (exclude_failure_reasons=[])")
        else:
            logger.info(f"Excluding episodes with reasons: {exclude_reasons}")

        logger.info(f"Extracting data from {input_dir} to {output_dir}")

        # 1. MCAP files discovery
        mcap_files = sorted(list(input_dir.rglob("*.mcap")))
        if not mcap_files:
            logger.error(f"No MCAP files found in {input_dir}")
            return None

        # Chunked extraction settings
        chunk_size = cfg.get("chunk_size", 500)
        total_mcaps = len(mcap_files)
        num_chunks = (total_mcaps + chunk_size - 1) // chunk_size
        
        logger.info(f"Total MCAP files: {total_mcaps}, Chunk size: {chunk_size}, Num chunks: {num_chunks}")

        from collections import Counter
        global_skipped_reasons = Counter()
        global_processed = 0
        global_samples = 0
        all_stats = []

        # 2. Process in chunks
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_mcaps)
            chunk_mcaps = mcap_files[start_idx:end_idx]
            
            batch_name = f"batch_{chunk_idx:04d}"
            scans_path = output_dir / f"{batch_name}_scans.npy"
            steers_path = output_dir / f"{batch_name}_steers.npy"
            accels_path = output_dir / f"{batch_name}_accelerations.npy"
            
            # Skip if batch already exists
            if scans_path.exists() and steers_path.exists() and accels_path.exists():
                logger.info(f"Skipping {batch_name} (already exists)")
                # Load stats from existing batch if available
                batch_stats_path = output_dir / f"{batch_name}_stats.json"
                if batch_stats_path.exists():
                    with open(batch_stats_path) as f:
                        batch_stats = json.load(f)
                        global_processed += batch_stats.get("processed_episodes", 0)
                        global_samples += batch_stats.get("total_samples", 0)
                        for reason, count in batch_stats.get("skipped_breakdown", {}).items():
                            global_skipped_reasons[reason] += count
                continue
            
            logger.info(f"Processing {batch_name}: MCAPファイル {start_idx}-{end_idx} ({len(chunk_mcaps)} files)")
            
            all_scans = []
            all_steers = []
            all_accels = []
            s_freq_list = []
            c_freq_list = []
            skipped_reasons = Counter()

            for mcap_file in chunk_mcaps:
                # Check result.json for success status
                result_json_path = mcap_file.parent / "result.json"
                if result_json_path.exists():
                    try:
                        with open(result_json_path) as f:
                            result_data = json.load(f)

                        if not result_data.get("success", False):
                            # Failed episode - check if we should skip it
                            if exclude_reasons is None:
                                # Exclude all failures
                                skipped_reasons["failed_unspecified"] += 1
                                continue

                            failure_reason = result_data.get("reason", "")
                            if failure_reason in exclude_reasons:
                                # This reason is in the exclude list
                                skipped_reasons[failure_reason] += 1
                                continue
                            # Else: include this failed episode
                    except Exception as e:
                        logger.warning(f"Failed to read result.json at {result_json_path}: {e}")
                        # If we can't read result.json, skip it unless we're including all
                        if exclude_reasons is None:
                            skipped_reasons["read_error"] += 1
                            continue

                result = self._extract_from_single_mcap(mcap_file)
                if result:
                    all_scans.append(result["scans"])
                    all_steers.append(result["steers"])
                    all_accels.append(result["accelerations"])
                    if "s_freq" in result:
                        s_freq_list.append(result["s_freq"])
                    if "c_freq" in result:
                        c_freq_list.append(result["c_freq"])

            if not all_scans:
                logger.warning(f"No data extracted for {batch_name}, skipping")
                continue

            scans = np.concatenate(all_scans, axis=0)
            steers = np.concatenate(all_steers, axis=0)
            accels = np.concatenate(all_accels, axis=0)

            # Save batch files
            np.save(scans_path, scans)
            np.save(steers_path, steers)
            np.save(accels_path, accels)

            # Update global counters
            processed_in_batch = len(chunk_mcaps) - sum(skipped_reasons.values())
            global_processed += processed_in_batch
            global_samples += len(scans)
            for reason, count in skipped_reasons.items():
                global_skipped_reasons[reason] += count

            # Save batch stats
            batch_stats = {
                "batch_name": batch_name,
                "mcap_range": [start_idx, end_idx],
                "processed_episodes": processed_in_batch,
                "total_samples": len(scans),
                "skipped_breakdown": dict(skipped_reasons),
            }
            with open(output_dir / f"{batch_name}_stats.json", "w") as f:
                json.dump(batch_stats, f, indent=2)
            
            all_stats.append(batch_stats)
            logger.info(f"Saved {batch_name}: {len(scans)} samples from {processed_in_batch} episodes")

        # 3. Calculate and save global statistics
        if global_samples == 0:
            logger.error("No data could be extracted from any MCAP file.")
            return None

        stats = {
            "dataset_overview": {
                "input_dir": str(input_dir),
                "total_episodes": total_mcaps,
                "processed_episodes": global_processed,
                "skipped_episodes": sum(global_skipped_reasons.values()),
                "total_samples": global_samples,
                "num_batches": num_chunks,
                "chunk_size": chunk_size,
            },
            "skipped_episodes_breakdown": dict(global_skipped_reasons),
            "batches": all_stats,
        }

        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        # 4. Save metadata (Lineage)
        self._save_metadata(output_dir, input_dir)

        logger.info(f"Successfully extracted {global_samples} samples in {num_chunks} batches.")
        logger.info(f"Dataset saved to {output_dir}")

        # 5. DVC Automation - Disabled as per user request
        # if cfg.dvc and cfg.dvc.auto_add:
        #     self._run_dvc_commands(output_dir, cfg.dvc.auto_push)

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

    def _clean_scan_array(self, scan_array: np.ndarray, max_range: float) -> np.ndarray:
        """Clean LiDAR scan data (replace NaNs/Infs and clip)."""
        if not isinstance(scan_array, np.ndarray):
            scan_array = np.array(scan_array, dtype=np.float32)
        cleaned = np.nan_to_num(scan_array, nan=0.0, posinf=max_range, neginf=0.0)
        cleaned = np.clip(cleaned, 0.0, max_range)
        return cleaned.astype(np.float32)

    def _synchronize_data(self, src_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
        """Find indices in target_times that correspond to src_times (nearest neighbor)."""
        if len(target_times) == 0:
            return np.array([], dtype=int)
        
        idx_sorted = np.searchsorted(target_times, src_times)
        idx_sorted = np.clip(idx_sorted, 0, len(target_times) - 1)
        
        prev_idx = np.clip(idx_sorted - 1, 0, len(target_times) - 1)
        
        time_diff_curr = np.abs(target_times[idx_sorted] - src_times)
        time_diff_prev = np.abs(target_times[prev_idx] - src_times)
        
        # Choose closer one
        use_prev = time_diff_prev < time_diff_curr
        final_indices = np.where(use_prev, prev_idx, idx_sorted)
        return final_indices

    def _extract_from_single_mcap(self, mcap_path: Path, cfg: DictConfig | None = None) -> dict[str, Any] | None:
        """Extract and sync data from one MCAP."""
        scans_list = []
        scan_times = []
        control_times = []
        control_data = []

        # Get topic names from config if available, otherwise defaults
        control_topic = "/control/command/control_cmd"
        scan_topic = "/sensing/lidar/scan"
        
        # Access topics via self.cfg if possible, or pass it down. 
        # Since _extract_from_single_mcap is called from _run_impl where self.cfg refers to Hydra cfg object...
        # Ideally we pass topics. But for member method we can use a class attribute or pass it.
        # Here we will check if 'experiment' in self.base_cfg (stored if we modify init) 
        # For now, let's assume we can rely on defaults or hardcode, 
        # BUT this method signature doesn't take config. 
        # Let's fix the call site in _run_impl or use hardcoded for now? 
        # Wait, I should update _run_impl to pass the topics or store cfg in self.
        
        # Update: In _run_impl, I will read topics from cfg and pass/use them.
        # But this method is replacing the WHOLE FILE content or chunk? 
        # I am replacing from 188 to 311.
        # I need to access the config. I'll modify the signature to accept topics map.
        
        pass 
        # Since I'm replacing the method, I can change signature, but I need to update the call site too.
        # To avoid multiple edits, I will assume the caller will be updated or I handle it.
        # Let's check _run_impl... oh wait I am only replacing the bottom methods.
        # I need to edit _run_impl too.
        
        # To be safe and minimal: I'll hardcode the retrieval from a class property or context if feasible?
        # No, better: I'll make this method use the topics I just added to YAML, 
        # assuming the caller (which I am not editing in this chunk) passes them? 
        # Actually I am editing a huge chunk. Let me check lines 188-311.
        # This covers _extract_from_single_mcap implementation.
        # I can change it to use self._topics dictionary if I set it in _run_impl.
        
        # However, _run_impl call site needs update.
        # Strategy:
        # 1. Update _run_impl to read topics and store in self.topics
        # 2. Update these methods to use self.topics
        
        # Let's stick to the current plan: Replace these methods first. 
        # I will use instance variable `self.topics` which I will initialize in _run_impl (next tool call).

        topic_control = self.topics.get("control", "/control/command/control_cmd")
        topic_scan = self.topics.get("scan", "/sensing/lidar/scan")
        target_topics = [topic_scan, topic_control]

        try:
            with open(mcap_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])

                for schema, channel, message in reader.iter_messages():
                    if channel.topic not in target_topics:
                        continue

                    msg = None
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
                        continue

                    if channel.topic == topic_scan:
                        ranges = None
                        if isinstance(msg, dict) and "ranges" in msg:
                            ranges = np.array(msg["ranges"], dtype=np.float32)
                        elif hasattr(msg, "ranges"):
                            ranges = np.array(msg.ranges, dtype=np.float32)

                        if ranges is not None:
                            # Use new robust cleaning
                            cleaned = self._clean_scan_array(ranges, 30.0)
                            scans_list.append(cleaned)
                            scan_times.append(message.log_time)

                    elif channel.topic == topic_control:
                        steer, accel, found = 0.0, 0.0, False
                        if isinstance(msg, dict):
                            # Handle both nested (lateral.steering...) and flat structures if needed, 
                            # but v3 script assumes specific structure. We follow v3.
                            if "lateral" in msg and "longitudinal" in msg:
                                steer = msg["lateral"].get("steering_tire_angle", 0.0)
                                accel = msg["longitudinal"].get("acceleration", 0.0)
                                found = True
                        else:
                            if hasattr(msg, "lateral"):
                                steer = msg.lateral.steering_tire_angle
                                accel = msg.longitudinal.acceleration
                                found = True
                        
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

        # Verify data frequency (Original logic preserved)
        s_times = np.array(scan_times, dtype=np.int64)
        c_times = np.array(control_times, dtype=np.int64)
        
        s_freq = None
        c_freq = None
        if len(s_times) > 1:
            s_diff = np.diff(s_times) / 1e9
            s_freq = 1.0 / np.mean(s_diff)
        if len(c_times) > 1:
            c_diff = np.diff(c_times) / 1e9
            c_freq = 1.0 / np.mean(c_diff)

        # Sync using NEW ROBUST LOGIC
        c_data = np.array(control_data, dtype=np.float32)
        idx = self._synchronize_data(s_times, c_times)
        
        synced_controls = c_data[idx]

        return {
            "scans": np.array(scans_list, dtype=np.float32),
            "steers": synced_controls[:, 0],
            "accelerations": synced_controls[:, 1],
            "s_freq": s_freq,
            "c_freq": c_freq,
        }
