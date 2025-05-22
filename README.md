# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn tools.

2.Load the Iris dataset and create a DataFrame with features and target.

3.Separate features (x) and labels (y), then split into training and testing sets.

4.Create and train an SGDClassifier on the training data.

5.Use the trained model to predict labels on the test data.

6.Calculate and print the accuracy of the model.

7.Generate and print the confusion matrix to assess classification performance.

8.Plot the true vs. predicted labels to visualize prediction distribution.

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: AMIRTHAVARSHINI V 
RegisterNumber: 212223040014
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
x = df.drop('target', axis=1)
y = df['target']
df.info()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sgd=SGDClassifier(max_iter=1000,tol=1e-3)
sgd.fit(x_train,y_train)
y_pred=sgd.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
confusion_mat=confusion_matrix(y_test,y_pred)
print("Confuison matrix: ",confusion_mat)
plt.scatter(y_test,y_pred)
```
## Output:
![prediction of iris species using SGD Classifier](sam.png)
![437749425-d9b945ba-1381-46ac-8702-7adb768782be](https://github.com/user-attachments/assets/ec41e819-e66c-4c48-94c6-e205be2783de)
![438917903-5418ae9c-ee29-47bb-a54d-680981b2fe63](https://github.com/user-attachments/assets/55647aec-67c0-415f-a695-93338b94cfd6)
![438918002-922fd1a3-f379-4437-a124-e68baa9c47c0](https://github.com/user-attachments/assets/3b3d21e6-3cbf-4fe8-b8c0-ce355879f314)
![438918365-e6a79929-c1b3-4afc-b77a-0745690d1fbb](https://github.com/user-attachments/assets/bdd9533b-03de-4fe5-a973-c86d61228383)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
