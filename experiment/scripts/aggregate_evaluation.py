#!/usr/bin/env python3
"""Aggregate results from evaluation runs.

Usage:
    uv run python experiment/scripts/aggregate_evaluation.py outputs/mlops/v8_20260104_202932/evaluation/standard
    uv run python experiment/scripts/aggregate_evaluation.py outputs/mlops/v8_20260104_202932/evaluation/debug
"""

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_result_files(eval_dir: Path) -> list[Path]:
    """Find all result.json files in evaluation subdirectories."""
    results = []
    # Search for all result.json files recursively
    for result_path in eval_dir.rglob("result.json"):
        results.append(result_path)
    return sorted(results)





def load_env_file() -> None:
    """Load .env file manually if it exists."""
    try:
        # Check current dir and parents for .env
        current_dir = Path.cwd()
        project_root = None
        for parent in [current_dir, *list(current_dir.parents)]:
            if (parent / ".env").exists():
                project_root = parent
                break
        
        if project_root:
            env_path = project_root / ".env"
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == "MCAP_BASE_URL" and "MCAP_BASE_URL" not in os.environ:
                            os.environ["MCAP_BASE_URL"] = value.strip()
    except Exception:
        pass


def aggregate_results(result_files: list[Path], eval_dir: Path) -> dict:
    """Aggregate all result files into a summary."""
    load_env_file()

    results = []
    reason_counter: Counter[str] = Counter()
    episodes_by_reason: dict[str, list[dict]] = {}
    base_url = os.getenv("MCAP_BASE_URL")
    # Fallback default host
    host = os.getenv("FOXGLOVE_HOST_IP", "127.0.0.1")

    for result_path in result_files:
        try:
            with open(result_path) as f:
                result = json.load(f)
                results.append(result)
                
                # Determine reason
                if result.get("success"):
                    reason = "goal_reached"
                else:
                    reason = result.get("reason") or "unknown"
                    if reason == "":
                        reason = "timeout"
                
                reason_counter[reason] += 1
                
                # Track which episodes have each reason
                if reason not in episodes_by_reason:
                    episodes_by_reason[reason] = []
                
                # Get episode directory name from path
                episode_dir = result_path.parent.name
                
                # Get the scenario name (e.g., "default", "no_obstacle")
                # Path structure: eval_dir/scenario/evaluation/episode_XXXX/result.json
                # Note: eval_dir in main is likely outputs/.../evaluation/standard
                try:
                    scenario = result_path.parent.parent.parent.name
                except Exception:
                    scenario = "unknown"
                
                episode_path = result_path.parent.relative_to(eval_dir)
                
                # Generate Foxglove URL dynamically to support remote viewing
                foxglove_url = ""
                try:
                    import urllib.parse
                    # Attempt to find outputs/ directory to calculate project-relative path
                    # We assume mcap is alongside result.json
                    mcap_path = result_path.parent / "simulation.mcap"
                    
                    # Search for 'outputs' in the path parents to find project root
                    project_root = None
                    for p in result_path.parents:
                        if p.name == "outputs":
                            project_root = p.parent
                            break
                    
                    if project_root:
                        full_rel_path = mcap_path.relative_to(project_root)
                        
                        if base_url:
                            base_url = base_url.rstrip("/")
                            mcap_url = f"{base_url}/{full_rel_path}"
                        elif "ts.net" in host:
                            mcap_url = f"https://{host}/{full_rel_path}"
                        else:
                            mcap_url = f"http://{host}:8080/{full_rel_path}"
                            
                        encoded_url = urllib.parse.quote(mcap_url, safe="")
                        foxglove_url = f"https://app.foxglove.dev/view?ds=remote-file&ds.url={encoded_url}"
                    else:
                         foxglove_url = result.get("foxglove_url", "")
                except Exception:
                     foxglove_url = result.get("foxglove_url", "")
                
                episodes_by_reason[reason].append({
                    "scenario": scenario,
                    "episode": episode_dir,
                    "seed": result.get("seed"),
                    "path": str(episode_path),
                    "foxglove": foxglove_url,
                    "metrics": result.get("metrics", {})
                })
        except Exception as e:
            logger.warning(f"Failed to read {result_path}: {e}")

    total = len(results)
    if total == 0:
        return None, None, None

    success_count = reason_counter.get("goal_reached", 0)
    success_rate = success_count / total

    reason_breakdown = {
        reason: {"count": count, "rate": count / total}
        for reason, count in reason_counter.items()
    }

    total_checkpoints = sum(r.get("metrics", {}).get("checkpoint_count", 0) for r in results)
    total_goals = sum(r.get("metrics", {}).get("goal_count", 0) for r in results)
    avg_checkpoints = total_checkpoints / total if total > 0 else 0

    # Get relative path from project root if possible
    try:
        output_dir_str = str(eval_dir.relative_to(Path.cwd()))
    except ValueError:
        output_dir_str = str(eval_dir)

    summary = {
        "evaluation_dir": output_dir_str,
        "total_episodes": total,
        "success_rate": success_rate,
        "reason_breakdown": reason_breakdown,
        "episodes_by_reason": episodes_by_reason,
        "aggregated_metrics": {
            "total_checkpoints": total_checkpoints,
            "total_goals": total_goals,
            "avg_checkpoints_per_episode": avg_checkpoints,
        },
    }

    return summary, reason_counter, episodes_by_reason


def plot_summary(output_dir: Path, reason_counter: Counter, total: int) -> None:
    """Generate summary plot."""
    try:
        import matplotlib.pyplot as plt

        color_map = {
            "goal_reached": "#4CAF50",
            "off_track": "#F44336",
            "collision": "#FF9800",
            "timeout": "#2196F3",
            "unknown": "#9E9E9E",
        }

        labels = list(reason_counter.keys())
        sizes = list(reason_counter.values())
        colors = [color_map.get(label, "#9C27B0") for label in labels]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart
        ax1.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*total)})",
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax1.set_title(f"Evaluation Outcomes (n={total})", fontsize=14, fontweight="bold")

        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_xlabel("Outcome", fontsize=12)
        ax2.set_title("Evaluation Outcome Distribution", fontsize=14, fontweight="bold")

        for bar, count in zip(bars, sizes):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()

        plot_path = output_dir / "evaluation_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Created plot: {plot_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
    except Exception as e:
        logger.warning(f"Failed to generate plot: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python experiment/scripts/aggregate_evaluation.py [evaluation_dir]")
        print("Example: uv run python experiment/scripts/aggregate_evaluation.py outputs/mlops/v8_20260104_202932/evaluation/standard")
        sys.exit(1)

    eval_dir = Path(sys.argv[1]).resolve()

    if not eval_dir.exists():
        logger.error(f"Directory not found: {eval_dir}")
        sys.exit(1)

    logger.info(f"Aggregating evaluation results from: {eval_dir}")

    result_files = find_result_files(eval_dir)
    logger.info(f"Found {len(result_files)} result files")

    if not result_files:
        logger.error("No result.json files found")
        sys.exit(1)

    summary, reason_counter, episodes_by_reason = aggregate_results(result_files, eval_dir)

    if summary is None:
        logger.error("Failed to aggregate results")
        sys.exit(1)

    # Save summary JSON
    summary_path = eval_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Created summary: {summary_path}")

    # Log to console
    logger.info(
        f"Summary: Total={summary['total_episodes']}, "
        f"Success={reason_counter.get('goal_reached', 0)} ({summary['success_rate']:.1%}), "
        f"Breakdown={dict(reason_counter)}"
    )

    # Generate plot
    plot_summary(eval_dir, reason_counter, summary["total_episodes"])

    print(f"\n✅ Aggregation complete! Output: {eval_dir}")


if __name__ == "__main__":
    main()
