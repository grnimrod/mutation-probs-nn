from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump


"""
Load in splits, write function to direct encode response variable
(for now, until I write custom function to calculate accuracy on vectors)
"""

X_train = np.load("/faststorage/project/MutationAnalysis/Nimrod/data/splits/X_train.npy")
y_train = np.load("/faststorage/project/MutationAnalysis/Nimrod/data/splits/y_train.npy")
X_val = np.load("/faststorage/project/MutationAnalysis/Nimrod/data/splits/X_val.npy")
y_val = np.load("/faststorage/project/MutationAnalysis/Nimrod/data/splits/y_val.npy")
X_test = np.load("/faststorage/project/MutationAnalysis/Nimrod/data/splits/X_test.npy")
y_test = np.load("/faststorage/project/MutationAnalysis/Nimrod/data/splits/y_test.npy")

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
timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
plt.savefig(f"/faststorage/project/MutationAnalysis/Nimrod/results/figures/conf_matrix_rf_{timestamp}.png")
plt.close()
print("Confusion matrix saved to file")

dump(rf, "/faststorage/project/MutationAnalysis/Nimrod/results/models/baseline_rf.joblib")