import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

reference = pd.read_csv("monitoring/reference_data.csv")
current = pd.read_csv("monitoring/current_data.csv")

# Compare only feature columns
feature_cols = [c for c in reference.columns if c.startswith("feature_")]

reference = reference[feature_cols]
current = current[feature_cols]

report = Report(metrics=[
    DataDriftPreset()
])

report.run(
    reference_data=reference,
    current_data=current
)

# Save HTML report
report.save_html("monitoring/evidently_report.html")

# Get results as dictionary
result = report.as_dict()

print(result)

drift_detected = result["metrics"][0]["result"]["dataset_drift"]

if drift_detected:
    print("🚨 Data Drift Detected")
else:
    print("✅ No Data Drift")