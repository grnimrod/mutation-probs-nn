from dataset import CustomDataset
from splits import create_splits

dataset = CustomDataset("./../data/15mer_A.tsv")

train_dataset, val_dataset, test_dataset = create_splits(dataset)