# Data

The application generates deterministic synthetic industrial telemetry at runtime. Run
`python scripts/generate_sensor_data.py` to write a local CSV here. Generated data is ignored by Git.

No real equipment, personal, or production data is included.

Required model features:

- `temperature_c`
- `pressure_bar`
- `vibration_mm_s`
- `flow_l_min`
- `motor_current_a`
- `acoustic_db`
- `rotation_rpm`
- `efficiency_ratio`

All values must be finite and numeric. An optional `sample_id` is copied into inference output.
Extra columns are ignored by model scoring.

Created by School of AI and School of QC.
