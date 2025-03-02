# Create subsets of the data that contain 5% of the full data
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


def subset_data(input_file):
    df = pd.read_csv(input_file, sep="\t")

    subset_df, _ = train_test_split(
        df,
        train_size=0.05,
        random_state=42,
        stratify=df["mut"]
    )

    filename = input_file.split(".")[-2]

    subset_df.to_csv(f"./../{filename}_subset.tsv", sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create subset of full data")
    parser.add_argument('input_file', type=str, help="Name of the input file")

    args = parser.parse_args()
    subset_data(args.input_file)