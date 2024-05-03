import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, BaggingClassifier, GradientBoostingClassifier

df=pd.read_csv("E:\churn\sentiment.csv",encoding="latin")
df1=df.fillna(0)
print(df1)

df1.columns=['sentimnets','id','date','null','name','sentence']
print(df1)
x=df1.drop(['date','id','null'],axis=1)
print(x)
y=df1['sentimnets']
print(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

# base_estimators = [
#     ('decision_tree', DecisionTreeClassifier()),
#     ('random_forest', RandomForestClassifier()),
#     ('bagging',BaggingClassifier()),
#     # ('AddaBoost',AdaBoostClassifier()),
#     # ('Gradi',GradientBoostingClassifier()),
#     # Add more base classifiers as needed
# ]

# rfc = StackingClassifier(estimators=base_estimators, final_estimator=(RandomForestClassifier()))
# rfc.fit(x_train, y_train)
# y_pred = rfc.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy Score :", accuracy * 100, "%")

model=LinearRegression()
model.fit(x_train,y_train)
yy=model.predict(x_test)
print(yy)

# rf=RandomForestClassifier()
# rf.fit(x_train,y_train)
# y_pred=rf.predict(x_test)

# accuracy_score=accuracy_score(y_test,yy>0.5)
# print("accc score",accuracy_score*100,"%")

#x_train.to_csv("D:\PycharmProjects\MachineLearning\Custommer_churn_rate\X_train.csv", index=False)
#x_test.to_csv("D:\PycharmProjects\MachineLearning\Custommer_churn_rate\X_test.csv", index=False)
#pd.DataFrame(y_train, columns=["churn"]).to_csv("D:\PycharmProjects\MachineLearning\Custommer_churn_rate\y_train.csv", index=False)
#pd.DataFrame(y_test, columns=["churn"]).to_csv("D:\PycharmProjects\MachineLearning\Custommer_churn_rate\y_test.csv", index=False)

# pd.DataFrame(y_pred, columns=["PredictedChurn"]).to_csv("predicted_Churn.csv", index=False)
#joblib.dump(yy, "E:\\churn\\internet_service_churn.pkl")
