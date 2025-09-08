import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
X, y = load_breast_cancer(return_X_y=True)
X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train1)
X_test = scaler.transform(X_test1)


clf = LogisticRegression()
clf.fit(X_train, y_train)
pred=clf.predict(X_test)
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))


s=SVC()
s.fit(X_train,y_train)
y_pred=s.predict(X_test)
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))


k=KNeighborsClassifier(5)
k.fit(X_train, y_train)
y_pred_1 = k.predict(X_test)
print(confusion_matrix(y_pred_1,y_test))
print(classification_report(y_pred_1,y_test))


d=DecisionTreeClassifier()
d.fit(X_train,y_train)
y_pred_2 = d.predict(X_test)
print(confusion_matrix(y_pred_2,y_test))
print(classification_report(y_pred_2,y_test))