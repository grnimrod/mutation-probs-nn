import pandas as pd


df = pd.read_csv("/faststorage/project/MutationAnalysis/Nimrod/data/processed/3mer_full_A.tsv", sep="\t")
df["mut_type"] = df["type"].str[-1]

mutation_counts = df.groupby(["context", "mut_type"]).size().unstack(fill_value=0)

mutation_counts = mutation_counts.div(mutation_counts.sum(axis=1), axis=0)

class KmerMutationModel:
    def __init__(self, prob_table):
        self.prob_table = prob_table # prob_table is a pandas dataframe
    
    def predict(self, kmer):
        return self.prob_table[kmer] if kmer in self.prob_table.index else pd.Series(0, index=self.prob_table.columns)


model = KmerMutationModel(mutation_counts)
pred = model.predict("AAA")
print(pred)