import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load
from collections import Counter
import matplotlib.pyplot as plt

from dataset import CustomDataset


dataset = CustomDataset("./../data/15mer_A.tsv") # TODO: convert one-hot encoding of response variable to single categorical encoding

train_indices = np.load("splits/train_indices.npy")
val_indices = np.load("splits/val_indices.npy")
test_indices = np.load("splits/test_indices.npy")

# Obtain train val test splits using saved indices, convert them to np arrays
def subset_to_numpy(dataset, indices):
    context = [dataset[i][0].numpy() for i in indices]
    res_mut = [np.argmax(dataset[i][1].numpy()) for i in indices]

    return np.array(context), np.array(res_mut)


X_train, y_train = subset_to_numpy(dataset, train_indices)
X_val, y_val = subset_to_numpy(dataset, val_indices)
X_test, y_test = subset_to_numpy(dataset, test_indices)

print("Distribution of mutation classes in training set: ", Counter(y_train))

# Flatten 3D arrays
X_train = X_train.reshape(len(X_train), -1)
X_val = X_val.reshape(len(X_val), -1)
X_test = X_test.reshape(len(X_test), -1)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

class_probs = rf.predict_proba(X_val)
print("Example probability distributions: ", class_probs[:5]) # Probability distribution of mutations of first five samples in X_val

y_pred = rf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Accuracy: ", accuracy)

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.savefig("./../conf_matrix_rf.png")
plt.close()
print("Confusion matrix saved to file")

dump(rf, "saved_models/baseline_rf.joblib")