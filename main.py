import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("E:\\churn\\internet_service_churn.csv")
ndf = df.fillna(0)
x = ndf.drop(['churn','id'], axis=1)
y = ndf["churn"]

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
#print(x_train,"xtrain")
#print(x_test,"xtest")
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# y_pred = lr.predict(x_test)
#-----

lor= LogisticRegression()
lor.fit(x_train,y_train)
y_pred1=lor.predict(x_test)

# knn= KNeighborsClassifier()
# knn.fit(x_train,y_train)
# y_pred22=knn.predict(x_test)

# svvm= svm()
# svvm.fit(x_train,y_train)
# y_pred23=svvm.predict(x_test)


accuracy = accuracy_score(y_test, y_pred1 > 0.5)
print("Accuracy Score logistic regression :", accuracy * 100, "%")

# accuracy = accuracy_score(y_test, y_pred1 > 0.5)
# print("Accuracy Score linear regression:", accuracy * 100, "%")

# accuracy = accuracy_score(y_test, y_pred22 > 0.5)
# print("Accuracy Score knn regression:", accuracy * 100, "%")

# accuracy = accuracy_score(y_test, y_pred23 > 0.5)
# print("Accuracy Score svm regression:", accuracy * 100, "%")


#----
#joblib.dump(lr, "E:\\churn\\internet_service_churn.pkl")