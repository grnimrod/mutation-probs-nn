from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import h5py
from joblib import dump


"""
Load in h5py splits, write function to direct encode response variable
(for now, until I write custom function to calculate accuracy on vectors)
"""

f = h5py.File("./../data/dataset.hdf5", "r")
X_train = f["group/X_train"]
y_train = f["group/y_train"][:] # Have to read the entire file into memory to be able to modify it later in the code
X_val = f["group/X_val"]
y_val = f["group/y_val"][:]
X_test = f["group/X_test"]
y_test = f["group/y_test"][:]

# Direct encode response variable (temporal solution)
def direct_encode(array):
    return np.argmax(array, axis=1).reshape(-1)

y_train = direct_encode(y_train)
y_val = direct_encode(y_val)
y_test = direct_encode(y_test)

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

f.close()