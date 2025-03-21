import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4]  # Only used with 'poly' kernel
}

svm_model = svm.SVC()

grid = GridSearchCV(svm_model, param_grid, refit=True, verbose=0)
grid.fit(x_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy}")

with open("best_svm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label="Actual", marker="o", linestyle="--", color="blue")
plt.plot(range(len(y_test)), y_pred, label="Predicted", marker="x", linestyle="-", color="red")
plt.title(f"SVM Classification (Best Model): Accuracy = {accuracy*100:.2f}%")
plt.xlabel("Test Data Index")
plt.ylabel("Class (0=Malignant, 1=Benign)")
plt.legend()
plt.show()

# Load the saved model (if needed later)
# with open("best_svm_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)
#     loaded_model.predict(x_test)
