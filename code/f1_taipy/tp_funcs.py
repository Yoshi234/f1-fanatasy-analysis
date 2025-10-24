import pandas as pd
import re

def get_features():
    data = []

    with open("../../results/baku/race_report.txt", "r") as file:
        lines = file.readlines()

    driverRef = None
    avg_fin = None

    for line in lines:
        line = line.strip()

        # Match section header
        header_match = re.match(r"\[DRIVER\]: (\w{3}) \[PRED\]: \[\[(\d+\.\d{1,2})\]\] \[BIAS\]: \[(\-?\d+\.\d{8})\]", line)
        if header_match:
            driverRef = header_match.group(1)
            avg_fin = float(header_match.group(2))
            continue

        # Skip section divider and column header
        if line.startswith("---") or "Input Value" in line or line == "":
            continue

        # Match feature lines
        feature_match = re.match(r"([^\s]+)\s+(-?\d+\.\d{6})\s+(-?\d+\.\d+)", line)
        if feature_match and driverRef is not None:
            feature = feature_match.group(1)
            input_value = float(feature_match.group(2))
            feature_score = float(feature_match.group(3))

            data.append({
                "driverRef": driverRef,
                "avg_fin": avg_fin,
                "feature": feature,
                "input_value": input_value,
                "feature_score": feature_score
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create a temporary column for absolute feature score
    df["abs_score"] = df["feature_score"].abs()

    # Group by driverRef and get top 3 rows by abs_score
    top_df = df.groupby("driverRef", group_keys=False).apply(lambda g: g.nlargest(3, "abs_score"))

    # Drop the temporary column if you don't need it
    top_df = top_df.drop(columns="abs_score")

    top_df = top_df.reset_index().drop('index', axis = 1)

    return top_df