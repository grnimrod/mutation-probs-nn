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
        return self.prob_table.loc[kmer] if kmer in self.prob_table.index else pd.Series(0, index=self.prob_table.columns)

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


class FullyConnectedNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_seq = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(4)
        )
    
    def forward(self, x):
        x = self.linear_relu_seq(x)
        return x
    
    def predict_proba(self, x):
        logits = self.linear_relu_seq(x)
        return F.softmax(logits, dim=-1)


class FCNNEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.flatten = nn.Flatten()

        self.linear_relu_seq = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(4)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.linear_relu_seq(x)
        return x
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class CombinedNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_seq = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.LazyLinear(64), nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.LazyLinear(4)
        )

    def forward(self, local_feature, expanded_feature):
        x = torch.concat([local_feature, expanded_feature], dim=1)
        x = self.linear_relu_seq(x)
        return x

    def predict_proba(self, context_vec, bin_id):
        logits = self.forward(context_vec, bin_id)
        return F.softmax(logits, dim=-1)


class LocalModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_seq = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.LazyLinear(32)
        )
    
    def forward(self, x):
        x = self.linear_relu_seq(x)
        return x


class ExpandedModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.flatten = nn.Flatten()

        self.linear_relu_seq = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.LazyLinear(32)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.linear_relu_seq(x)
        return x


class CombinedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.seq(x)


class MixedInputModel(nn.Module):
    def __init__(self, local_model, expanded_model, mixed_model):
        super().__init__()

        self.local_model = local_model
        self.expanded_model = expanded_model
        self.mixed_model = mixed_model

    def forward(self, x):
        x_local, x_region = x

        local_out = self.local_model(x_local)
        expanded_out = self.expanded_model(x_region)

        x_combined = torch.cat((local_out, expanded_out), dim=1)
        return self.mixed_model(x_combined)
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)