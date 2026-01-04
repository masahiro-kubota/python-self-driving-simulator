#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict

"""
MLOps Pipeline Automation Script for Obstacle Avoidance

This script automates:
1. Data Collection (TF/RS, Train/Val)
2. Aggregation (Collection Summary)
3. Combined Feature Extraction (MCAP -> NumPy)
4. Training (TinyLidarNet)
5. Evaluation (Standard & Debug)

Usage:
    # Full pipeline execution
    uv run python experiment/scripts/run_mlops_pipeline_obstacle.py \
        --version v8 \
        --tf-train 1000 --tf-val 200 \
        --rs-train 500 --rs-val 100

    # Skip steps for faster iteration
    uv run python experiment/scripts/run_mlops_pipeline_obstacle.py \
        --version v8 \
        --skip-collection --skip-extraction --skip-training \
        --model-path /path/to/best_model.npy

Arguments:
    --version: Dataset version identifier (e.g., v8)
    --tf-train: Track Forward training episodes (default: 1000)
    --tf-val: Track Forward validation episodes (default: 200)
    --rs-train: Random Start training episodes (default: 500)
    --rs-val: Random Start validation episodes (default: 100)
    --skip-collection: Skip data collection step
    --skip-extraction: Skip feature extraction step
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


class MLOpsPipeline:
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_root = Path.cwd()
        
        self.dataset_version = args.version
        
        # Base directory for this entire run
        self.run_base_dir = self.project_root / "outputs" / "mlops" / f"{self.dataset_version}_{self.timestamp}"
        self.collection_base_dir = self.run_base_dir / "collection"
        
        # Processed data directories (Final destination for training)
        self.processed_data_root = self.project_root / "data" / "processed"
        self.train_data_dir = self.processed_data_root / f"train_{self.dataset_version}"
        self.val_data_dir = self.processed_data_root / f"val_{self.dataset_version}"
        
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
            ("track_forward", "train", self.args.tf_train, 0),
            ("track_forward", "val", self.args.tf_val, 100000),
            ("random_start", "train", self.args.rs_train, 200000),
            ("random_start", "val", self.args.rs_val, 300000),
        ]
        
        for exp_type, split, count, base_seed in jobs:
            if count <= 0:
                logger.info(f"Skipping {exp_type} {split} (count=0)")
                continue
                
            logger.info(f"--- Collecting {exp_type} {split} (N={count}) ---")
            
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
        """Step 3: Feature Extraction (Combined)."""
        logger.info("=== Step 3: Feature Extraction (Combined) ===")
        
        exclude_reasons = "'[off_track,collision,unknown]'"
        
        for split in ["train", "val"]:
            output_dir = self.train_data_dir if split == "train" else self.val_data_dir
            
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
            self.run_command(cmd, description=f"Combined Extraction {split}")

    def run_training(self):
        """Step 4: Training."""
        logger.info("=== Step 4: Training ===")
        
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
        """Step 5: Evaluation."""
        logger.info("=== Step 5: Evaluation ===")
        
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
        
        # Collect extraction stats
        for split in ["train", "val"]:
            output_dir = self.train_data_dir if split == "train" else self.val_data_dir
            stats_file = output_dir / "stats.json"
            if stats_file.exists():
                try:
                    with open(stats_file) as f:
                        key = f"{split}_extraction_stats"
                        summary["intermediate_results"][key] = json.load(f)
                        logger.info(f"Included extraction stats for {split}")
                except Exception as e:
                    logger.warning(f"Failed to load {stats_file}: {e}")
        
        self.run_base_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Saved pipeline summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Obstacle Avoidance MLOps Pipeline")
    parser.add_argument("--version", type=str, required=True, help="Dataset version (e.g., v8)")
    
    # Episode counts
    parser.add_argument("--tf-train", type=int, default=1000, help="Track Forward Train episodes")
    parser.add_argument("--tf-val", type=int, default=200, help="Track Forward Val episodes")
    parser.add_argument("--rs-train", type=int, default=500, help="Random Start Train episodes")
    parser.add_argument("--rs-val", type=int, default=100, help="Random Start Val episodes")
    
    # Flags
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    
    parser.add_argument("--model-path", type=str, help="Reuse existing model for evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute commands")
    parser.add_argument("--continue-on-error", action="store_true", help="Don't stop on command failure")

    args = parser.parse_args()
    
    pipeline = MLOpsPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
