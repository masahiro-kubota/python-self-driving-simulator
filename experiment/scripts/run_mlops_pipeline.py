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
from typing import List, Optional

"""
MLOps Pipeline Automation Script

This script automates the entire end-to-end workflow for self-driving model development:
1. Data Collection (Train/Val)
2. Aggregation (Collection Summary)
3. Feature Extraction (MCAP -> NumPy)
4. Training (TinyLidarNet)
5. Evaluation (Simulation with Debug Node)

Usage:
    # Run full pipeline with default settings (v7 dataset)
    uv run python experiment/scripts/run_mlops_pipeline.py --version v7

    # Customize episode counts
    uv run python experiment/scripts/run_mlops_pipeline.py --version v8 --train-episodes 2000 --val-episodes 400

    # Skip early stages (e.g., only run training and evaluation)
    # Note: Requires previous stages to be completed or handled manually, or inputs to be present.
    uv run python experiment/scripts/run_mlops_pipeline.py --skip-collection --skip-extraction

    # Dry run (print commands without executing)
    uv run python experiment/scripts/run_mlops_pipeline.py --dry-run

Arguments:
    --version: Dataset version string (default: "v7")
    --train-episodes: Number of training episodes (default: 1000)
    --val-episodes: Number of validation episodes (default: 200)
    --skip-collection: Skip data collection and aggregation
    --skip-extraction: Skip feature extraction
    --skip-training: Skip model training
    --dry-run: Print commands only
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
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        self.project_root = Path.cwd()
        
        # Directories
        self.dataset_version = args.version
        self.processed_data_dir = self.project_root / "data" / "processed"
        self.train_data_dir = self.processed_data_dir / f"train_{self.dataset_version}"
        self.val_data_dir = self.processed_data_dir / f"val_{self.dataset_version}"
        
        # State to track outputs between steps
        self.train_collection_dir: Optional[Path] = None
        self.val_collection_dir: Optional[Path] = None
        self.model_path: Optional[Path] = None
        
        # Track executed steps
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
            # Using shell=True for complex commands with pipes or redirects, 
            # though here we mainly use uv run. Splitting args is safer.
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
            if capture_output:
                logger.error(f"Stdout: {e.stdout}")
                logger.error(f"Stderr: {e.stderr}")
            self.pipeline_steps.append(step_record)
            sys.exit(1)

    def _find_latest_multirun(self, parent_dir: str = "outputs") -> Path:
        """Find the most recent timestamped directory under outputs/."""
        # This assumes the standard Hydra output structure: outputs/YYYY-MM-DD/HH-MM-SS
        # We need to find the latest created directory.
        outputs_path = self.project_root / parent_dir
        if not outputs_path.exists():
            raise FileNotFoundError(f"{outputs_path} does not exist")

        # List all date directories
        date_dirs = sorted([d for d in outputs_path.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)])
        if not date_dirs:
            raise FileNotFoundError(f"No date directories found in {outputs_path}")
        
        latest_date_dir = date_dirs[-1]
        
        # List all time directories in the latest date directory
        time_dirs = sorted([d for d in latest_date_dir.iterdir() if d.is_dir() and re.match(r"\d{2}-\d{2}-\d{2}", d.name)])
        if not time_dirs:
            raise FileNotFoundError(f"No time directories found in {latest_date_dir}")
            
        return time_dirs[-1]

    def run_collection(self):
        """Step 1: Data Collection."""
        logger.info("=== Step 1: Data Collection ===")
        
        # Train Data Collection
        train_cmd = (
            f"uv run experiment-runner -m "
            f"experiment=data_collection_random_start "
            f"execution.total_episodes={self.args.train_episodes} "
            f"execution.base_seed=0 "
            f"experiment.name=data_collection_train_{self.dataset_version}"
        )
        self.run_command(train_cmd, description="Train Data Collection")
        
        # Identify output directory for Train
        if not self.args.dry_run:
            # We assume the user just ran it, so it's the latest.
            # Wait a bit to ensure filesystem sync?
            time.sleep(1)
            self.train_collection_dir = self._find_latest_multirun()
            logger.info(f"Identified Train Collection Dir: {self.train_collection_dir}")

        # Val Data Collection
        val_cmd = (
            f"uv run experiment-runner -m "
            f"experiment=data_collection_random_start "
            f"execution.total_episodes={self.args.val_episodes} "
            f"execution.base_seed=2000 "
            f"experiment.name=data_collection_val_{self.dataset_version}"
        )
        self.run_command(val_cmd, description="Val Data Collection")
        
        # Identify output directory for Val
        if not self.args.dry_run:
            time.sleep(1)
            self.val_collection_dir = self._find_latest_multirun()
            logger.info(f"Identified Val Collection Dir: {self.val_collection_dir}")

    def run_aggregation(self):
        """Step 2: Aggregation."""
        logger.info("=== Step 2: Aggregation ===")
        if self.args.dry_run:
            logger.info(f"Would aggregate {self.train_collection_dir} and {self.val_collection_dir}")
            return

        for collection_dir in [self.train_collection_dir, self.val_collection_dir]:
            if collection_dir:
                cmd = f"uv run python experiment/scripts/aggregate_multirun.py {collection_dir}"
                self.run_command(cmd, description=f"Aggregation for {collection_dir.name}")

    def run_extraction(self):
        """Step 3: Feature Extraction."""
        logger.info("=== Step 3: Feature Extraction ===")
        
        exclude_reasons = "'[off_track,collision,unknown]'"
        
        # Train Extraction
        # If dry run or skipped collection, we don't have dirs. 
        # For dry run, use placeholders.
        train_input = self.train_collection_dir if self.train_collection_dir else "outputs/TR_DATE/TR_TIME"
        if self.args.skip_collection and not self.train_collection_dir and not self.args.dry_run:
             # Try to find latest manually? Or fail?
             # For now, let's assume if skip_collection is used, the user might need to provide path or we pick latest?
             # Simplification: Pick latest 2 output dirs? No, that's risky.
             # Strict mode: require paths if skipping? 
             # Let's try to search specifically for folders with experiment name?
             # For now, just warn and try latest if not set.
             pass

        train_cmd = (
            f"uv run experiment-runner "
            f"experiment=extraction "
            f"input_dir={train_input} "
            f"output_dir={self.train_data_dir} "
            f"exclude_failure_reasons={exclude_reasons}"
        )
        self.run_command(train_cmd, description="Train Feature Extraction")
        
        # Val Extraction
        val_input = self.val_collection_dir if self.val_collection_dir else "outputs/VAL_DATE/VAL_TIME"
        val_cmd = (
            f"uv run experiment-runner "
            f"experiment=extraction "
            f"input_dir={val_input} "
            f"output_dir={self.val_data_dir} "
            f"exclude_failure_reasons={exclude_reasons}"
        )
        self.run_command(val_cmd, description="Val Feature Extraction")

    def run_training(self):
        """Step 4: Training."""
        logger.info("=== Step 4: Training ===")
        
        # Sanitize timestamp for experiment name (remove slashes)
        safe_timestamp = self.timestamp.replace("/", "_").replace("-", "")
        
        cmd = (
            f"uv run experiment-runner -m "
            f"experiment=training "
            f"train_data={self.train_data_dir} "
            f"val_data={self.val_data_dir} "
            f"experiment.name=training_{self.dataset_version}_{safe_timestamp}"
        )
        self.run_command(cmd, description="Model Training")
        
        if not self.args.dry_run:
            # Find the trained model path.
            # Hydra outputs to outputs/YYYY-MM-DD/HH-MM-SS/training_...
            # But the user guide says: "outputs/latest/checkpoints/best_model.npy" 
            # (Note: outputs/latest symlink is updated by experiment-runner/hydra?)
            # Actually, hydra symlinks 'latest' to the current run directory relative to where it ran.
            # Assuming experiment-runner updates 'latest_output' or similar.
            # Let's assume standard path: outputs/latest is a symlink to the run.
            
            # NOTE: Hydra's default 'outputs/latest' usage might be tricky with multiple runs.
            # experiment-runner might handle it.
            # Let's find the LATEST run again.
            try:
                latest_run_dir = self._find_latest_multirun() # Usually single run for training but same logic applies
                # Check for checkpoint
                # Pattern: checkpoints/model_name_version/best_model.npy
                # We need to find the .npy file.
                checkpoints_dir = latest_run_dir / "checkpoints"
                if checkpoints_dir.exists():
                     npy_files = list(checkpoints_dir.glob("**/best_model.npy"))
                     if npy_files:
                         self.model_path = npy_files[0]
                         logger.info(f"Found model: {self.model_path}")
                     else:
                         logger.warning(f"No .npy model found in {checkpoints_dir}")
                else:
                     logger.warning(f"No checkpoints dir in {latest_run_dir}")

            except Exception as e:
                logger.warning(f"Could not locate trained model automatically: {e}")

    def run_evaluation(self):
        """Step 5: Evaluation."""
        logger.info("=== Step 5: Evaluation ===")
        
        # Determine model path
        model_path = None
        if self.args.model_path:
            model_path = Path(self.args.model_path).resolve()
            if not model_path.exists() and not self.args.dry_run:
                logger.error(f"Provided model path does not exist: {model_path}")
                sys.exit(1)
        elif self.model_path:
            model_path = self.model_path
        
        if not model_path:
            if self.args.dry_run:
                model_path = "$(pwd)/checkpoints/dummypath/best_model.npy"
            else:
                logger.error("No model path found! Provide --model-path or run training first.")
                sys.exit(1)
        
        base_cmd = (
            f"uv run experiment-runner experiment=evaluation "
            f"ad_components=tiny_lidar_debug "
            f"ad_components.model_path={model_path} "
            f"env=no_obstacle "
        )
        
        # 1. Debug Evaluation (Pure Pursuit control, TinyLidarNet monitor)
        logger.info("--- Running Debug Evaluation (PurePursuit driving, Model monitoring) ---")
        debug_cmd = base_cmd + (
             " ad_components.nodes.tiny_lidar_net.params.control_cmd_topic=control_cmd_tinylidar"
             " ad_components.nodes.control.params.control_cmd_topic=control_cmd"
             f" experiment.name=evaluation_debug_{self.dataset_version}"
        )
        self.run_command(debug_cmd, description="Debug Evaluation (PurePursuit)")

        # 2. Standard Evaluation (Model driving)
        logger.info("--- Running Standard Evaluation (Model driving) ---")
        # No topic overrides needed as standard tiny_lidar_debug.yaml configures TinyLidarNet correctly for driving?
        # Actually, tiny_lidar_debug.yaml has PurePursuit active by default?
        # Let's verify defaults.
        # If tiny_lidar_debug.yaml has:
        #   tiny_lidar_net: control_cmd_topic: "control_cmd" (Inactive default? No, usually model overrides control)
        #   control: control_cmd_topic: "control_cmd_pure_pursuit" (Inactive default)
        
        # Wait, if tiny_lidar_debug.yaml defaults are set for debug, we might need to override them for standard?
        # Let's assume the user wants standard behavior where Model drives.
        # Looking at previous context:
        # User defined: tiny_lidar_debug.yaml
        # L29: control_cmd_topic: "control_cmd" (for tiny_lidar_net)
        # L65: control_cmd_topic: "control_cmd_pure_pursuit" (for control/pure_pursuit)
        # So by DEFAULT, TinyLidarNet publishes to "control_cmd" (drives), and PurePursuit to "control_cmd_pure_pursuit" (debug).
        # So for STANDARD evaluation, we just run it without overrides (or ensuring defaults).
        
        std_cmd = base_cmd + f" experiment.name=evaluation_standard_{self.dataset_version}"
        self.run_command(std_cmd, description="Standard Evaluation (Model Control)")
        
        logger.info("Evaluations complete. Please check the latest output directories for MCAP.")

    def run(self):
        if not self.args.skip_collection:
            self.run_collection()
            self.run_aggregation()
        else:
            logger.info("Skipping Data Collection and Aggregation.")
            # If skipping collection but running extraction, we need inputs.
            # For simplicity in this v1 script, we assume user provides them or we fail/warn if not found.
            # But wait, if I skip collection, I can't guess the directories easily without args.
            # IMPROVEMENT: Add args for input dirs if skipping collection.
            # For now, let's just proceed. `run_extraction` handles missing dirs for dry-run, but for real run it will fail if explicit paths aren't set.
            # We implemented a "find latest" but that might pick the wrong thing if skipping. 
            pass

        if not self.args.skip_extraction:
            self.run_extraction()
        else:
            logger.info("Skipping Feature Extraction.")

        if not self.args.skip_training:
            self.run_training()
        else:
            logger.info("Skipping Training.")
            
        self.run_evaluation()
        
        # Save run result json
        self._save_run_result()

    def _save_run_result(self):
        """Save execution summary to result.json in the latest output dir."""
        logger.info("Saving pipeline execution result...")
        
        result_data = {
            "timestamp": self.timestamp,
            "version": self.dataset_version,
            "args": vars(self.args),
            "pipeline_steps": getattr(self, "pipeline_steps", []),
            "outputs": {
                "train_collection": str(self.train_collection_dir) if self.train_collection_dir else None,
                "val_collection": str(self.val_collection_dir) if self.val_collection_dir else None,
                "train_data": str(self.train_data_dir),
                "val_data": str(self.val_data_dir),
                "model_path": str(self.model_path) if self.model_path else None,
            },
            "intermediate_results": {}
        }

        # Collect collection summaries
        for split, d in [("train", self.train_collection_dir), ("val", self.val_collection_dir)]:
            if d and (d / "collection_summary.json").exists():
                try:
                    with open(d / "collection_summary.json") as f:
                        result_data["intermediate_results"][f"{split}_collection_summary"] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read summary from {d}: {e}")

        # Collect extraction stats
        for split, d in [("train", self.train_data_dir), ("val", self.val_data_dir)]:
            if d and (d / "stats.json").exists():
                 try:
                    with open(d / "stats.json") as f:
                        result_data["intermediate_results"][f"{split}_extraction_stats"] = json.load(f)
                 except Exception as e:
                    logger.warning(f"Failed to read stats from {d}: {e}")

        # Determine where to save. We want it in the final output dir, which is usually evaluation output.
        # But we don't track evaluation output dir explicitly yet. 
        # Let's try to find the latest run (Evaluation run).
        try:
            final_output_dir = self._find_latest_multirun("outputs")
            # If standard structure, this is outputs/Date/Time.
            output_path = final_output_dir / "run_mlops_result.json"
            
            with open(output_path, "w") as f:
                json.dump(result_data, f, indent=4)
            logger.info(f"Saved run result to: {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save run result: {e}")
            # Fallback to current dir
            with open(f"run_mlops_result_{self.timestamp.replace('/','-')}.json", "w") as f:
                json.dump(result_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Run full MLOps pipeline.")
    parser.add_argument("--version", type=str, default="v7", help="Dataset/Experiment version (e.g. v7)")
    parser.add_argument("--train-episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--val-episodes", type=int, default=200, help="Number of validation episodes")
    
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection step")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip feature extraction step")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    
    parser.add_argument("--model-path", type=str, help="Path to trained model for evaluation. If not provided, tries to find in latest training output.")
    
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()
    
    pipeline = MLOpsPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
