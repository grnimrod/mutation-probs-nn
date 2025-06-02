import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import log_loss
import pickle


class KmerCountsModel:
    """
    Simple count-based probabilistic model.
    Requires one-hot data to be decoded into labels
    """
    def __init__(self):
        self.prob_table = None

    def fit(self, X_train, y_train):
        train_df = pd.DataFrame({
            "context": X_train,
            "mut_type": y_train
        })
        mutation_counts = train_df.groupby(["context", "mut_type"]).size().unstack(fill_value=0)
        mutation_counts = mutation_counts.div(mutation_counts.sum(axis=1), axis=0)
        self.prob_table = mutation_counts

    def predict(self, kmer):
        # Should only be able to run if fit() method was already ran
        if self.prob_table is None:
            raise ValueError("Model is not trained yet.")
        return self.prob_table.loc[kmer] if kmer in self.prob_table.index else self.prob_table.mean(axis=0)

    def evaluate(self, X_test, y_test):
        y_true = []
        y_pred = []
        for kmer, mut in zip(X_test, y_test):
            probs = self.predict(kmer)
            y_true.append(mut)
            y_pred.append(probs)
        pred_df = pd.DataFrame(y_pred)
        return log_loss(y_true, pred_df, labels=pred_df.columns)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.prob_table, f)

    @classmethod # This is needed in order to access the class itself to instantiate it
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            prob_table = pickle.load(f)
        model = cls()
        model.prob_table = prob_table
        return model


class ModularModel(nn.Module):
    def __init__(self, use_avg_mut=False, use_bin_id_embed=False, use_bin_id_norm=False, num_bins=None, embed_dim=None):
        super().__init__()
        self.use_avg_mut = use_avg_mut
        self.use_bin_id_embed = use_bin_id_embed
        self.use_bin_id_norm = use_bin_id_norm

        if use_bin_id_embed:
            self.embedding = nn.Embedding(num_embeddings=num_bins, embedding_dim=embed_dim)
            self.flatten = nn.Flatten()
        
        self.linear_relu_seq = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            nn.LazyLinear(64), nn.ReLU(),
            nn.LazyLinear(4)
        )
    
    def forward(self, local_context, avg_mut=None, bin_id=None, bin_id_norm=None):
        inputs = [local_context]

        if self.use_avg_mut:
            inputs.append(avg_mut)
        
        if self.use_bin_id_embed:
            assert self.embedding is not None, "Embedding layer not initialized. Did you forget to pass num_bins and embed_dim?"
            embedded = self.embedding(bin_id)
            flat = self.flatten(embedded)
            inputs.append(flat)
        
        if self.use_bin_id_norm:
            inputs.append(bin_id_norm)
        
        x = torch.cat(inputs, dim=-1)
        x = self.linear_relu_seq(x)
        return x
    
    def predict_proba(self, local_context, avg_mut=None, bin_id=None, bin_id_norm=None):
        logits = self.forward(local_context, avg_mut=avg_mut, bin_id=bin_id, bin_id_norm=bin_id_norm)
        return F.softmax(logits, dim=-1)