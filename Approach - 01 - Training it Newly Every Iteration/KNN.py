import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

plt.figure(figsize=(10,6))
plt.plot(range(len(y_test)), y_test, label="Actual", marker="o", linestyle="--", color="blue")
plt.plot(range(len(y_test)), y_pred, label="Predicted", marker="x", linestyle="-", color="red")
plt.title(f"KNN Classification: Accuracy = {accuracy*100:.2f}%")
plt.xlabel("Test Data Index")
plt.ylabel("Class (0=Malignant, 1=Benign)")
plt.legend()
plt.show()
