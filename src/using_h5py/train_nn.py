import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import h5py


# With torch model: load in h5py splits, create custom Dataset class, create DataLoaders

f = h5py.File("./../data/dataset.hdf5", "r")
X_train = f["group/X_train"][:]
y_train = f["group/y_train"][:]
X_val = f["group/X_val"][:]
y_val = f["group/y_val"][:]
X_test = f["group/X_test"][:]
y_test = f["group/y_test"][:]

print(type(X_train))
print(f"Before converting: {X_train[:3]}")
print(f"After converting:{np.array(X_train[:3])}")
print(type(X_train[0]))
X_train = torch.as_tensor(X_train)

# dataset = CustomDataset("./../data/15mer_A.tsv")

# train_dataset, val_dataset, test_dataset = create_splits(dataset) # TODO: somehow X_train and y_train should be separable while also keeping site position and mutation-nonmutation info

# def get_variables_for_training(dataset):
#     context = [torch.as_tensor(dataset[i][0]) for i in range(len(dataset))]
#     res_mut = [torch.as_tensor(dataset[i][1]) for i in range(len(dataset))]
#     return context, res_mut

# print(type(train_dataset[0]))

# X_train, y_train = get_variables_for_training(train_dataset)
# print(X_train[0])
# print(X_train.shape)

# lr = 0.01
# epochs = 10
# bs = 64

# # Wrap DataLoader iterator around our custom dataset(s)
# train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=bs*2, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=bs*2, shuffle=False)

# # Choose device
# device = "cpu" # CPU for now, if working with full dataset, include option to utilize GPU

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_seq = nn.Sequential(
#             nn.Linear(in_features=15*4, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=4)
#         )
#         self.softmax = nn.Softmax()
    
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear_relu_seq(x)
#         x = self.softmax(x)
#         return x


# # model = NeuralNetwork().to(device)
# # print(model)

# def get_model():
#     model = NeuralNetwork()
#     return model, optim.SGD(model.parameters(), lr=lr)


# # model, opt = get_model()
# # for epoch in range(epochs):
# #     for i in range():
# #         start_i = i

f.close()