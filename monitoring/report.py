import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

reference_data = pd.read_csv("monitoring/reference_data.csv")
current_data = pd.read_csv("monitoring/current_data.csv")

feature_cols = [c for c in reference_data.columns if c.startswith("feature_")]

report = Report(metrics=[
    DataDriftPreset()
])

report.run(
    reference_data=reference_data[feature_cols],
    current_data=current_data[feature_cols]
)

report.save_html("monitoring/evidently_report.html")

print("Report generated successfully!")