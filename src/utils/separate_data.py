# Separate dataset into two depending on reference nucleotide
import pandas as pd
import argparse


def separate_data(input_data):
    df = pd.read_csv(input_data, sep="\t")

    for nucl in df["type"].str[0].unique():
        data = df[df["type"].str[0] == nucl]

        filename = f"15mer_{nucl}.tsv"

        data.to_csv(f"./../data/{filename}", sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate dataset based on reference nucleotides")
    parser.add_argument('input_file', type=str, help="Name of the input file to be separated")

    args = parser.parse_args()
    separate_data(args.input_file)