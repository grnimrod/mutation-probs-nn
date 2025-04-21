import pandas as pd
import argparse


def extract_kmer(input_file, kmer_length):
    """
    Extract central k-mer of specified length
    """

    assert kmer_length % 2 == 1, "k-mer must be of odd length"

    df = pd.read_csv(input_file, sep="\t")

    sample_context = df.iloc[0]["context"] # See first example to obtain length for calculations later
    context_length = len(sample_context)
    middle_index = context_length // 2

    df["context"] = df["context"].str[(middle_index - (kmer_length // 2)):(middle_index + (kmer_length // 2) + 1)]

    df.to_csv(f"/faststorage/project/MutationAnalysis/Nimrod/data/raw/{kmer_length}mer_full.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract k-mer of specified length")
    parser.add_argument("input_file", type=str, help="Location of the input file to be used")
    parser.add_argument("kmer_length", type=int, help="Length of k-mer to be extracted")

    args = parser.parse_args()
    extract_kmer(args.input_file, args.kmer_length)