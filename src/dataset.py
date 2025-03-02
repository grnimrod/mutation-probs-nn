import torch
from torch.utils.data import Dataset
import pandas as pd


alphabet = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
    }

class CustomDataset(Dataset):
    def __init__(self, tsv_data):
        self.data = pd.read_csv(tsv_data, sep="\t")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Context around reference nucleotide
        kmer = row['context']
        kmer_encoded = torch.tensor([alphabet[nucl] for nucl in kmer], dtype=torch.float32)

        # Nucleotide that reference nucleotide is mutated into
        res_nucl = row['type'][-1]
        res_nucl_encoded = torch.tensor(alphabet[res_nucl], dtype=torch.float32)

        # Mutation or non-mutation info
        mut = row['mut']

        # Site position
        site = row['pos']

        return kmer_encoded, res_nucl_encoded, mut, site
    
    def get_mut_labels(self):
        # Create list of mutations used for stratified sampling
        return self.data['mut'].tolist()