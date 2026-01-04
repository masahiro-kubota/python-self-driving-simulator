#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict

"""
MLOps Pipeline Automation Script for Obstacle Avoidance with Data Merging

This script automates:
1. Data Collection (TF/RS, Train/Val) - NEW data only
2. Aggregation (Collection Summary)
3. Feature Extraction (MCAP -> NumPy) - NEW data only
4. Data Merging (Combine with existing extracted dataset)
5. Training (TinyLidarNet)
6. Evaluation (Standard & Debug)

Usage:
    # Merge new TF data with existing v8 dataset
    uv run python experiment/scripts/run_mlops_pipeline_obstacle_merge_data.py \
        --version v9 \
        --tf-train 16000 --tf-val 4000 \
        --rs-train 0 --rs-val 0 \
        --merge-train-dir data/processed/train_v8 \
        --merge-val-dir data/processed/val_v8 \
        --tf-train-base-seed 8000 --tf-val-base-seed 102000

    # Dry run to check commands
    uv run python experiment/scripts/run_mlops_pipeline_obstacle_merge_data.py \
        --version v9 \
        --tf-train 16000 --tf-val 4000 \
        --merge-train-dir data/processed/train_v8 \
        --merge-val-dir data/processed/val_v8 \
        --dry-run

Arguments:
    --version: New dataset version identifier (e.g., v9)
    --tf-train: Track Forward training episodes to collect (default: 0)
    --tf-val: Track Forward validation episodes to collect (default: 0)
    --rs-train: Random Start training episodes to collect (default: 0)
    --rs-val: Random Start validation episodes to collect (default: 0)
    --tf-train-base-seed: Base seed for TF train (default: 0)
    --tf-val-base-seed: Base seed for TF val (default: 100000)
    --rs-train-base-seed: Base seed for RS train (default: 200000)
    --rs-val-base-seed: Base seed for RS val (default: 300000)
    --merge-train-dir: Path to existing train dataset to merge with
    --merge-val-dir: Path to existing val dataset to merge with
    --skip-collection: Skip data collection step
    --skip-extraction: Skip feature extraction step
    --skip-merge: Skip data merging step
    --skip-training: Skip training step
    --skip-evaluation: Skip evaluation step
    --model-path: Path to existing model for evaluation (optional)
    --dry-run: Print commands without executing
    --continue-on-error: Continue pipeline even if a step fails
"""

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mlops_pipeline")


class MLOpsPipelineMerge:
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_root = Path.cwd()
        
        self.dataset_version = args.version
        
        # Base directory for this entire run
        self.run_base_dir = self.project_root / "outputs" / "mlops" / f"{self.dataset_version}_{self.timestamp}"
        self.collection_base_dir = self.run_base_dir / "collection"
        
        # Processed data directories
        self.processed_data_root = self.project_root / "data" / "processed"
        
        # Temporary extraction directories (for new data only)
        self.new_train_data_dir = self.processed_data_root / f"train_{self.dataset_version}_new"
        self.new_val_data_dir = self.processed_data_root / f"val_{self.dataset_version}_new"
        
        # Final merged directories (for training)
        self.train_data_dir = self.processed_data_root / f"train_{self.dataset_version}"
        self.val_data_dir = self.processed_data_root / f"val_{self.dataset_version}"
        
        # Existing datasets to merge with
        self.merge_train_dir = Path(args.merge_train_dir).resolve() if args.merge_train_dir else None
        self.merge_val_dir = Path(args.merge_val_dir).resolve() if args.merge_val_dir else None
        
        # State tracking
        self.collection_dirs: Dict[str, Dict[str, Path]] = {
            "train": {},
            "val": {}
        }
        self.model_path: Optional[Path] = None
        self.pipeline_steps: List[dict] = []

    def run_command(self, command: str, description: str = "", capture_output: bool = False) -> Optional[str]:
        """Run a shell command."""
        logger.info(f"Running [{description}]: {command}")
        
        step_record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "command": command,
            "status": "pending"
        }
        
        if self.args.dry_run:
            step_record["status"] = "dry_run"
            self.pipeline_steps.append(step_record)
            return None

        try:
            # Use shell=True for complex commands if needed, but here we split
            cmd_args = shlex.split(command)
            
            if capture_output:
                result = subprocess.run(
                    cmd_args, 
                    check=True, 
                    text=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                step_record["status"] = "success"
                self.pipeline_steps.append(step_record)
                return result.stdout.strip()
            else:
                subprocess.run(cmd_args, check=True)
                step_record["status"] = "success"
                self.pipeline_steps.append(step_record)
                return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            step_record["status"] = "failed"
            step_record["exit_code"] = e.returncode
            self.pipeline_steps.append(step_record)
            if not self.args.continue_on_error:
                sys.exit(1)
            return None

    def _find_latest_multirun(self, parent_dir: Path) -> Path:
        """Find the most recent Hydra multirun directory."""
        if not parent_dir.exists():
            raise FileNotFoundError(f"{parent_dir} does not exist")

        # Let's look for date-pattern directories first, then latest time.
        date_dirs = sorted([d for d in parent_dir.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)])
        if date_dirs:
            latest_date_dir = date_dirs[-1]
            time_dirs = sorted([d for d in latest_date_dir.iterdir() if d.is_dir() and re.match(r"\d{2}-\d{2}-\d{2}", d.name)])
            if time_dirs:
                return time_dirs[-1]
        
        # Fallback: just return the latest created directory
        all_dirs = sorted([d for d in parent_dir.glob("**/*") if d.is_dir()], key=lambda x: x.stat().st_mtime)
        if all_dirs:
            return all_dirs[-1]
            
        raise FileNotFoundError(f"Could not find any output directory in {parent_dir}")

    def run_collection(self):
        """Step 1: Data Collection for both TF and RS."""
        logger.info("=== Step 1: Data Collection ===")
        
        jobs = [
            ("track_forward", "train", self.args.tf_train, self.args.tf_train_base_seed),
            ("track_forward", "val", self.args.tf_val, self.args.tf_val_base_seed),
            ("random_start", "train", self.args.rs_train, self.args.rs_train_base_seed),
            ("random_start", "val", self.args.rs_val, self.args.rs_val_base_seed),
        ]
        
        for exp_type, split, count, base_seed in jobs:
            if count <= 0:
                logger.info(f"Skipping {exp_type} {split} (count=0)")
                continue
                
            logger.info(f"--- Collecting {exp_type} {split} (N={count}, seed={base_seed}) ---")
            
            job_dir = (self.collection_base_dir / split / exp_type).resolve()
            
            cmd = (
                f"uv run experiment-runner -m "
                f"experiment=data_collection_{exp_type} "
                f"execution.total_episodes={count} "
                f"execution.base_seed={base_seed} "
                f"experiment.name=col_{exp_type}_{split}_{self.dataset_version} "
                f"hydra.sweep.dir={job_dir}"
            )
            
            self.run_command(cmd, description=f"Collection {exp_type} {split}")
            
            if not self.args.dry_run:
                # Since we set hydra.sweep.dir, the job_dir itself should be the multirun root.
                self.collection_dirs[split][exp_type] = job_dir

    def run_aggregation(self):
        """Step 2: Aggregation."""
        logger.info("=== Step 2: Aggregation ===")
        for split in ["train", "val"]:
            for exp_type, path in self.collection_dirs[split].items():
                if path.exists():
                    cmd = f"uv run python experiment/scripts/aggregate_multirun.py {path}"
                    self.run_command(cmd, description=f"Aggregation {exp_type} {split}")
                else:
                    logger.warning(f"Path not found for aggregation: {path}")

    def run_extraction(self):
        """Step 3: Feature Extraction (Combined) - Extract NEW data only."""
        logger.info("=== Step 3: Feature Extraction (New Data Only) ===")
        
        exclude_reasons = "'[off_track,collision,unknown]'"
        
        for split in ["train", "val"]:
            # Extract to temporary directory for new data
            output_dir = self.new_train_data_dir if split == "train" else self.new_val_data_dir
            
            # Point to the split directory (contains both TF and RS subdirs)
            input_dir = (self.collection_base_dir / split).resolve()
            
            if self.args.dry_run:
                input_dir = f"outputs/mlops/VERSION_TIME/collection/{split}"
            
            cmd = (
                f"uv run experiment-runner "
                f"experiment=extraction "
                f"input_dir={input_dir} "
                f"output_dir={output_dir} "
                f"exclude_failure_reasons={exclude_reasons}"
            )
            self.run_command(cmd, description=f"Extraction {split} (new data)")

    def run_merge(self):
        """Step 4: Merge new extracted data with existing dataset."""
        logger.info("=== Step 4: Data Merging ===")
        
        import numpy as np
        
        for split in ["train", "val"]:
            new_data_dir = self.new_train_data_dir if split == "train" else self.new_val_data_dir
            merge_dir = self.merge_train_dir if split == "train" else self.merge_val_dir
            output_dir = self.train_data_dir if split == "train" else self.val_data_dir
            
            if self.args.dry_run:
                logger.info(f"Dry run: Would merge {merge_dir} + {new_data_dir} -> {output_dir}")
                step_record = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "description": f"Merge {split} data",
                    "command": f"merge({merge_dir}, {new_data_dir}) -> {output_dir}",
                    "status": "dry_run"
                }
                self.pipeline_steps.append(step_record)
                continue
            
            logger.info(f"--- Merging {split} data ---")
            logger.info(f"  Existing: {merge_dir}")
            logger.info(f"  New:      {new_data_dir}")
            logger.info(f"  Output:   {output_dir}")
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing data
            existing_scans = None
            existing_steers = None
            existing_accelerations = None
            existing_count = 0
            
            if merge_dir and merge_dir.exists():
                existing_scans_path = merge_dir / "scans.npy"
                existing_steers_path = merge_dir / "steers.npy"
                existing_accelerations_path = merge_dir / "accelerations.npy"
                
                if existing_scans_path.exists() and existing_steers_path.exists() and existing_accelerations_path.exists():
                    existing_scans = np.load(existing_scans_path)
                    existing_steers = np.load(existing_steers_path)
                    existing_accelerations = np.load(existing_accelerations_path)
                    existing_count = len(existing_scans)
                    logger.info(f"  Loaded {existing_count} existing samples from {merge_dir}")
                else:
                    logger.warning(f"  Missing scans.npy, steers.npy, or accelerations.npy in {merge_dir}")
            else:
                logger.info(f"  No existing dataset to merge for {split}")
            
            # Load new data
            new_scans = None
            new_steers = None
            new_accelerations = None
            new_count = 0
            
            if new_data_dir.exists():
                new_scans_path = new_data_dir / "scans.npy"
                new_steers_path = new_data_dir / "steers.npy"
                new_accelerations_path = new_data_dir / "accelerations.npy"
                
                if new_scans_path.exists() and new_steers_path.exists() and new_accelerations_path.exists():
                    new_scans = np.load(new_scans_path)
                    new_steers = np.load(new_steers_path)
                    new_accelerations = np.load(new_accelerations_path)
                    new_count = len(new_scans)
                    logger.info(f"  Loaded {new_count} new samples from {new_data_dir}")
                else:
                    logger.warning(f"  Missing scans.npy, steers.npy, or accelerations.npy in {new_data_dir}")
            else:
                logger.warning(f"  New data directory does not exist: {new_data_dir}")
            
            # Merge
            if existing_scans is not None and new_scans is not None:
                merged_scans = np.concatenate([existing_scans, new_scans], axis=0)
                merged_steers = np.concatenate([existing_steers, new_steers], axis=0)
                merged_accelerations = np.concatenate([existing_accelerations, new_accelerations], axis=0)
            elif existing_scans is not None:
                merged_scans = existing_scans
                merged_steers = existing_steers
                merged_accelerations = existing_accelerations
            elif new_scans is not None:
                merged_scans = new_scans
                merged_steers = new_steers
                merged_accelerations = new_accelerations
            else:
                logger.error(f"  No data to merge for {split}!")
                continue
            
            # Save merged data
            np.save(output_dir / "scans.npy", merged_scans)
            np.save(output_dir / "steers.npy", merged_steers)
            np.save(output_dir / "accelerations.npy", merged_accelerations)
            
            total_count = len(merged_scans)
            logger.info(f"  Merged: {existing_count} + {new_count} = {total_count} samples")
            
            # Create merge stats
            stats = {
                "existing_samples": existing_count,
                "new_samples": new_count,
                "total_samples": total_count,
                "existing_dir": str(merge_dir) if merge_dir else None,
                "new_dir": str(new_data_dir),
                "output_dir": str(output_dir),
                "scans_shape": list(merged_scans.shape),
                "steers_shape": list(merged_steers.shape),
                "accelerations_shape": list(merged_accelerations.shape),
            }
            
            with open(output_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            step_record = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": f"Merge {split} data",
                "command": f"merge({merge_dir}, {new_data_dir}) -> {output_dir}",
                "status": "success",
                "stats": stats
            }
            self.pipeline_steps.append(step_record)

    def run_training(self):
        """Step 5: Training."""
        logger.info("=== Step 5: Training ===")
        
        safe_timestamp = self.timestamp.replace("_", "")
        # Use run_base_dir for training output too
        training_out = self.run_base_dir / "training"
        
        cmd = (
            f"uv run experiment-runner -m "
            f"experiment=training "
            f"train_data={self.train_data_dir} "
            f"val_data={self.val_data_dir} "
            f"experiment.name=train_{self.dataset_version}_{safe_timestamp} "
            f"hydra.sweep.dir={training_out}"
        )
        self.run_command(cmd, description="Model Training")
        
        if not self.args.dry_run:
            try:
                # 1. First check inside the dedicated mlops run directory
                checkpoints_dirs = list(training_out.glob("**/checkpoints"))
                if checkpoints_dirs:
                    latest_cp_dir = sorted(checkpoints_dirs, key=lambda x: x.stat().st_mtime)[-1]
                else:
                    # 2. Fallback: Search globally in outputs/YYYY-MM-DD/HH-MM-SS
                    outputs_dir = self.project_root / "outputs"
                    latest_hydra_dir = self._find_latest_multirun(outputs_dir)
                    latest_cp_dir = latest_hydra_dir / "checkpoints"
                    logger.info(f"Falling back to global outputs: {latest_cp_dir}")

                if latest_cp_dir.exists():
                    npy_files = list(latest_cp_dir.glob("**/best_model.npy"))
                    if npy_files:
                        self.model_path = npy_files[-1].resolve()
                        logger.info(f"Detected Model: {self.model_path}")
                    else:
                        logger.warning(f"No .npy model found in {latest_cp_dir}")
                else:
                    logger.warning(f"No checkpoints dir found.")
            except Exception as e:
                logger.warning(f"Could not automatically locate model: {e}")

    def run_evaluation(self):
        """Step 6: Evaluation."""
        logger.info("=== Step 6: Evaluation ===")
        
        model_path = self.args.model_path or self.model_path
        if not model_path:
            if self.args.dry_run:
                model_path = "/path/to/best_model.npy"
            else:
                logger.error("No model path found for evaluation!")
                return

        eval_out = self.run_base_dir / "evaluation"
        
        # Standard Evaluation (Autonomous driving with TinyLidarNet)
        logger.info("--- Running Standard Evaluation ---")
        
        # 1a. Standard Evaluation: no_obstacle environment
        std_no_obs_cmd = (
            f"uv run experiment-runner experiment=evaluation "
            f"ad_components=tiny_lidar_debug "
            f"ad_components.model_path={model_path} "
            f"env=no_obstacle "
            f"execution.num_episodes=1 "
            f"experiment.name=eval_std_no_obs_{self.dataset_version} "
            f"postprocess.foxglove.auto_open=false "
            f"hydra.run.dir={eval_out}/standard/no_obstacle"
        )
        self.run_command(std_no_obs_cmd, description="Standard Eval (no_obstacle)")

        # 1b. Standard Evaluation: default environment (env unspecified)
        std_default_cmd = (
            f"uv run experiment-runner "
            f"experiment=evaluation "
            f"ad_components=tiny_lidar_debug "
            f"ad_components.model_path={model_path} "
            f"execution.num_episodes=5 "
            f"experiment.name=eval_std_default_{self.dataset_version} "
            f"postprocess.foxglove.auto_open=false "
            f"hydra.run.dir={eval_out}/standard/default"
        )
        self.run_command(std_default_cmd, description="Standard Eval (default)")

        # Debug Evaluation (Monitoring with TinyLidarNet while Pure Pursuit drives)
        logger.info("--- Running Debug Evaluation ---")
        
        # 2a. Debug Evaluation: no_obstacle environment
        dbg_no_obs_cmd = (
            f"uv run experiment-runner experiment=evaluation "
            f"ad_components=tiny_lidar_debug "
            f"ad_components.model_path={model_path} "
            f"env=no_obstacle "
            f"execution.num_episodes=1 "
            f"experiment.name=eval_debug_no_obs_{self.dataset_version} "
            f"ad_components.nodes.tiny_lidar_net.params.control_cmd_topic=control_cmd_tinylidar "
            f"ad_components.nodes.control.params.control_cmd_topic=control_cmd "
            f"postprocess.foxglove.auto_open=false "
            f"hydra.run.dir={eval_out}/debug/no_obstacle"
        )
        self.run_command(dbg_no_obs_cmd, description="Debug Eval (no_obstacle)")

        # 2b. Debug Evaluation: default environment (env unspecified)
        dbg_default_cmd = (
            f"uv run experiment-runner experiment=evaluation "
            f"ad_components=tiny_lidar_debug "
            f"ad_components.model_path={model_path} "
            f"execution.num_episodes=1 "
            f"experiment.name=eval_debug_default_{self.dataset_version} "
            f"ad_components.nodes.tiny_lidar_net.params.control_cmd_topic=control_cmd_tinylidar "
            f"ad_components.nodes.control.params.control_cmd_topic=control_cmd "
            f"postprocess.foxglove.auto_open=false "
            f"hydra.run.dir={eval_out}/debug/default"
        )
        self.run_command(dbg_default_cmd, description="Debug Eval (default)")

        # Aggregate Standard Evaluation Results
        logger.info("--- Aggregating Standard Evaluation Results ---")
        std_eval_dir = eval_out / "standard"
        if not self.args.dry_run and std_eval_dir.exists():
            aggregate_cmd = f"uv run python experiment/scripts/aggregate_evaluation.py {std_eval_dir}"
            self.run_command(aggregate_cmd, description="Aggregate Standard Eval")

    def run(self):
        if not self.args.skip_collection:
            self.run_collection()
            self.run_aggregation()
        
        if not self.args.skip_extraction:
            self.run_extraction()
        
        if not self.args.skip_merge:
            self.run_merge()
            
        if not self.args.skip_training:
            self.run_training()
        else:
            if not self.args.model_path:
                logger.info("Training skipped, searching for the latest model globally...")
                try:
                    outputs_dir = self.project_root / "outputs"
                    latest_hydra_dir = self._find_latest_multirun(outputs_dir)
                    latest_cp_dir = latest_hydra_dir / "checkpoints"
                    npy_files = list(latest_cp_dir.glob("**/best_model.npy"))
                    if npy_files:
                        self.model_path = npy_files[-1].resolve()
                        logger.info(f"Detected Model: {self.model_path}")
                except Exception as e:
                    logger.warning(f"Could not find model for skipped training: {e}")
            
        if not self.args.skip_evaluation:
            self.run_evaluation()
            
        self._save_summary()

    def _save_summary(self):
        summary_path = self.run_base_dir / "pipeline_summary.json"
        if self.args.dry_run:
            logger.info(f"Dry run: Would save summary to {summary_path}")
            return
            
        summary = {
            "version": self.dataset_version,
            "timestamp": self.timestamp,
            "args": vars(self.args),
            "merge_info": {
                "merge_train_dir": str(self.merge_train_dir) if self.merge_train_dir else None,
                "merge_val_dir": str(self.merge_val_dir) if self.merge_val_dir else None,
                "new_train_dir": str(self.new_train_data_dir),
                "new_val_dir": str(self.new_val_data_dir),
                "output_train_dir": str(self.train_data_dir),
                "output_val_dir": str(self.val_data_dir),
            },
            "steps": self.pipeline_steps,
            "model": str(self.model_path) if self.model_path else None,
            "intermediate_results": {}
        }
        
        # Collect summaries from each Job
        for split, types in self.collection_dirs.items():
            for exp_type, path in types.items():
                summary_file = path / "collection_summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file) as f:
                            key = f"{split}_{exp_type}_summary"
                            summary["intermediate_results"][key] = json.load(f)
                            logger.info(f"Included aggregation summary for {split}/{exp_type}")
                    except Exception as e:
                        logger.warning(f"Failed to load {summary_file}: {e}")
        
        # Collect merge stats
        for split in ["train", "val"]:
            output_dir = self.train_data_dir if split == "train" else self.val_data_dir
            stats_file = output_dir / "stats.json"
            if stats_file.exists():
                try:
                    with open(stats_file) as f:
                        key = f"{split}_merge_stats"
                        summary["intermediate_results"][key] = json.load(f)
                        logger.info(f"Included merge stats for {split}")
                except Exception as e:
                    logger.warning(f"Failed to load {stats_file}: {e}")
        
        self.run_base_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Saved pipeline summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Obstacle Avoidance MLOps Pipeline with Data Merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge new TF data with existing v8 dataset
  uv run python experiment/scripts/run_mlops_pipeline_obstacle_merge_data.py \\
      --version v9 \\
      --tf-train 16000 --tf-val 4000 \\
      --merge-train-dir data/processed/train_v8 \\
      --merge-val-dir data/processed/val_v8 \\
      --tf-train-base-seed 8000 --tf-val-base-seed 102000

  # Skip to training (data already merged)
  uv run python experiment/scripts/run_mlops_pipeline_obstacle_merge_data.py \\
      --version v9 \\
      --skip-collection --skip-extraction --skip-merge
        """
    )
    parser.add_argument("--version", type=str, required=True, help="New dataset version (e.g., v9)")
    
    # Episode counts
    parser.add_argument("--tf-train", type=int, default=0, help="Track Forward Train episodes to collect")
    parser.add_argument("--tf-val", type=int, default=0, help="Track Forward Val episodes to collect")
    parser.add_argument("--rs-train", type=int, default=0, help="Random Start Train episodes to collect")
    parser.add_argument("--rs-val", type=int, default=0, help="Random Start Val episodes to collect")
    
    # Base seeds (to avoid overlap with existing data)
    parser.add_argument("--tf-train-base-seed", type=int, default=0, help="Base seed for TF train (default: 0)")
    parser.add_argument("--tf-val-base-seed", type=int, default=100000, help="Base seed for TF val (default: 100000)")
    parser.add_argument("--rs-train-base-seed", type=int, default=200000, help="Base seed for RS train (default: 200000)")
    parser.add_argument("--rs-val-base-seed", type=int, default=300000, help="Base seed for RS val (default: 300000)")
    
    # Merge directories
    parser.add_argument("--merge-train-dir", type=str, default=None, 
                        help="Path to existing train dataset to merge with")
    parser.add_argument("--merge-val-dir", type=str, default=None,
                        help="Path to existing val dataset to merge with")
    
    # Flags
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    
    parser.add_argument("--model-path", type=str, help="Reuse existing model for evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute commands")
    parser.add_argument("--continue-on-error", action="store_true", help="Don't stop on command failure")

    args = parser.parse_args()
    
    pipeline = MLOpsPipelineMerge(args)
    pipeline.run()


if __name__ == "__main__":
    main()
