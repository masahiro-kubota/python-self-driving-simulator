#!/usr/bin/env python3
"""Aggregate results from Hydra multirun data collection.

Usage:
    uv run python scripts/aggregate_multirun.py outputs/2026-01-04/13-59-54
    uv run python scripts/aggregate_multirun.py outputs/latest  # if symlink exists
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_result_files(multirun_dir: Path) -> list[Path]:
    """Find all result.json files in multirun subdirectories."""
    results = []
    for job_dir in sorted(multirun_dir.iterdir()):
        if not job_dir.is_dir() or not job_dir.name.isdigit():
            continue
        # Search for result.json in nested structure
        for result_path in job_dir.rglob("result.json"):
            results.append(result_path)
    return results


def aggregate_results(result_files: list[Path], multirun_dir: Path) -> dict:
    """Aggregate all result files into a summary."""
    results = []
    reason_counter: Counter[str] = Counter()
    episodes_by_reason: dict[str, list[dict]] = {}

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
                job_dir = result_path.parent.parent.parent.parent.name  # job number
                episode_path = result_path.parent.relative_to(multirun_dir)
                mcap_rel_path = episode_path / "simulation.mcap"
                
                # Generate Foxglove URL with full path including date/time
                import urllib.parse
                # multirun_dir is like outputs/2026-01-04/14-24-35, we need to get date/time parts
                date_time_path = multirun_dir.relative_to(multirun_dir.parent.parent)
                mcap_url = f"http://127.0.0.1:8080/outputs/{date_time_path}/{mcap_rel_path}"
                encoded_url = urllib.parse.quote(mcap_url, safe="")
                foxglove_url = f"https://app.foxglove.dev/view?ds=remote-file&ds.url={encoded_url}"
                
                episodes_by_reason[reason].append({
                    "job": job_dir,
                    "episode": episode_dir,
                    "seed": result.get("seed"),
                    "path": str(episode_path),
                    "foxglove": foxglove_url
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
    avg_checkpoints = total_checkpoints / total

    summary = {
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
        ax1.set_title(f"Episode Outcomes (n={total})", fontsize=14, fontweight="bold")

        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_xlabel("Outcome", fontsize=12)
        ax2.set_title("Episode Outcome Distribution", fontsize=14, fontweight="bold")

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

        plot_path = output_dir / "collection_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Created plot: {plot_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
    except Exception as e:
        logger.warning(f"Failed to generate plot: {e}")

def find_latest_multirun(outputs_dir: Path) -> Path | None:
    """Find the most recent multirun directory in outputs."""
    if not outputs_dir.exists():
        return None
    
    # Look for directories with date/time pattern
    latest = None
    latest_mtime = 0
    
    for date_dir in outputs_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir():
                continue
            # Check if it looks like a multirun (has numbered subdirs)
            has_numbered_dirs = any(
                d.name.isdigit() for d in time_dir.iterdir() if d.is_dir()
            )
            if has_numbered_dirs:
                mtime = time_dir.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest = time_dir
    
    return latest


def main():
    if len(sys.argv) < 2:
        # Auto-detect latest multirun directory
        outputs_dir = Path.cwd() / "outputs"
        multirun_dir = find_latest_multirun(outputs_dir)
        if multirun_dir is None:
            print("Usage: uv run python experiment/scripts/aggregate_multirun.py [multirun_dir]")
            print("       If no directory specified, auto-detects latest from outputs/")
            print("\nNo multirun directories found in outputs/")
            sys.exit(1)
        logger.info(f"Auto-detected latest multirun: {multirun_dir}")
    else:
        multirun_dir = Path(sys.argv[1]).resolve()

    if not multirun_dir.exists():
        logger.error(f"Directory not found: {multirun_dir}")
        sys.exit(1)

    logger.info(f"Aggregating results from: {multirun_dir}")

    result_files = find_result_files(multirun_dir)
    logger.info(f"Found {len(result_files)} result files")

    if not result_files:
        logger.error("No result.json files found")
        sys.exit(1)

    summary, reason_counter, episodes_by_reason = aggregate_results(result_files, multirun_dir)

    if summary is None:
        logger.error("Failed to aggregate results")
        sys.exit(1)

    # Save summary JSON
    summary_path = multirun_dir / "collection_summary.json"
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
    plot_summary(multirun_dir, reason_counter, summary["total_episodes"])

    print(f"\n✅ Aggregation complete! Output: {multirun_dir}")


if __name__ == "__main__":
    main()
