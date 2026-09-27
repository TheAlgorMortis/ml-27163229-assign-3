import json
from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("processed-data")
OUTPUT_DIR = Path("splits")
DEVELOPMENT_FRACTION = 0.80
TEST_FRACTION = 0.20


def find_timestamp_column(df):
    preferred = ["timestamp", "datetime", "date", "time", "Date Time", "Date"]

    for name in preferred:
        if name in df.columns:
            parsed = pd.to_datetime(df[name], errors="coerce")
            if parsed.notna().all():
                return name

    first_column = df.columns[0]
    parsed = pd.to_datetime(df[first_column], errors="coerce")

    if parsed.notna().all():
        return first_column

    raise ValueError(
        "Could not identify a valid timestamp column. "
        f"Available columns: {list(df.columns)}"
    )


def split_dataset(dataset_name):
    input_path = PROCESSED_DIR / f"{dataset_name}.csv"


    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"{input_path} is empty")

    timestamp_column = find_timestamp_column(df)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="raise")
    df = df.sort_values(timestamp_column).reset_index(drop=True)

    if df[timestamp_column].duplicated().any():
        raise ValueError(f"{dataset_name} contains duplicate timestamps")

    split_index = int(len(df) * DEVELOPMENT_FRACTION)

    if split_index <= 0 or split_index >= len(df):
        raise ValueError(
            f"Invalid split index for {dataset_name}: {split_index}"
        )

    development = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    dataset_output_dir = OUTPUT_DIR / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    development_path = dataset_output_dir / "development.csv"
    test_path = dataset_output_dir / "test.csv"
    metadata_path = dataset_output_dir / "split.json"

    development.to_csv(development_path, index=False)
    test.to_csv(test_path, index=False)

    metadata = {
        "dataset": dataset_name,
        "source_file": str(input_path),
        "timestamp_column": timestamp_column,
        "split_method": "chronological_80_20",
        "split_index": int(split_index),
        "total_rows": int(len(df)),
        "development_rows": int(len(development)),
        "test_rows": int(len(test)),
        "development_fraction": float(len(development) / len(df)),
        "test_fraction": float(len(test) / len(df)),
        "full_start": df[timestamp_column].iloc[0].isoformat(),
        "full_end": df[timestamp_column].iloc[-1].isoformat(),
        "development_start": development[timestamp_column].iloc[0].isoformat(),
        "development_end": development[timestamp_column].iloc[-1].isoformat(),
        "test_start": test[timestamp_column].iloc[0].isoformat(),
        "test_end": test[timestamp_column].iloc[-1].isoformat(),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(
        f"{dataset_name}: "
        f"development={len(development):,} "
        f"({metadata['development_fraction']:.1%}), "
        f"test={len(test):,} "
        f"({metadata['test_fraction']:.1%}), "
        f"test_start={metadata['test_start']}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, 6):
        split_dataset(f"ds{i}")

    print(f"\nSaved splits to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
