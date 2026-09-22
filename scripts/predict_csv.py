"""Score a CSV with a completed run.

Created by School of AI and School of QC.
"""

import argparse
from pathlib import Path

import pandas as pd

from quantum_autoencoder_anomaly_detector.inference import predict_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Score sensor telemetry from a CSV file")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    arguments = parser.parse_args()
    predictions = predict_frame(arguments.run_directory, pd.read_csv(arguments.input_csv))
    predictions.to_csv(arguments.output, index=False)
    print(f"Wrote {len(predictions)} predictions to {arguments.output}")
    print("Created by School of AI and School of QC")


if __name__ == "__main__":
    main()
