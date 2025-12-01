"""Utility script to download training data from MLflow."""

import argparse
from pathlib import Path

import mlflow


def download_training_data(run_id: str, output_dir: str = "data/raw/downloaded") -> None:
    """Download training data from MLflow run.

    Args:
        run_id: MLflow run ID
        output_dir: Output directory for downloaded data
    """
    client = mlflow.tracking.MlflowClient()

    # Get run info
    run = client.get_run(run_id)
    print(f"Run: {run.info.run_name}")
    print(f"Experiment: {run.data.tags.get('mlflow.experimentName', 'Unknown')}")

    # Download training data artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        artifact_path = client.download_artifacts(
            run_id, "training_data", dst_path=str(output_path)
        )
        print(f"\nDownloaded training data to: {artifact_path}")

        # List downloaded files
        data_dir = Path(artifact_path)
        if data_dir.exists():
            files = list(data_dir.glob("*.json"))
            print(f"Found {len(files)} data files:")
            for f in files:
                print(f"  - {f.name}")
    except Exception as e:
        print(f"Error downloading training data: {e}")

    # Download config if available
    try:
        config_path = client.download_artifacts(run_id, "config", dst_path=str(output_path))
        print(f"\nDownloaded config to: {config_path}")
    except Exception:
        print("\nNo config file found in this run")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download training data from MLflow")
    parser.add_argument("run_id", help="MLflow run ID")
    parser.add_argument(
        "--output-dir",
        default="data/raw/downloaded",
        help="Output directory (default: data/raw/downloaded)",
    )

    args = parser.parse_args()
    download_training_data(args.run_id, args.output_dir)


if __name__ == "__main__":
    main()
