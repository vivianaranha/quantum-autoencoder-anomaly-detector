"""Generate a reproducible demonstration dataset.

Created by School of AI and School of QC.
"""

from pathlib import Path

from quantum_autoencoder_anomaly_detector.data import generate_sensor_telemetry


def main() -> None:
    destination = Path("data/synthetic_sensor_telemetry.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_sensor_telemetry(sample_count=900, anomaly_rate=0.12, random_seed=42)
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame)} rows to {destination}")
    print("Created by School of AI and School of QC")


if __name__ == "__main__":
    main()
