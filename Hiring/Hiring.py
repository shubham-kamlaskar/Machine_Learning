import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

data = {
    "experience":[np.nan,np.nan,"five","two","seven","three","ten","eleven"],
    "test_score":[8,8,6,10,9,7,np.nan,7],
    "interview_score":[9,6,7,10,6,10,7,8],
    "salary":[50000,45000,60000,65000,70000,62000,72000,80000]
}

hr = pd.DataFrame(data)


exp = {"five":5,"two":2,"seven":7,"three":3,"ten":10,"eleven":11}
hr["experience"] = hr["experience"].map(exp)


fill = math.floor(hr["test_score"].mean())
hr["test_score"]=hr["test_score"].fillna(fill)

exp1 = hr["experience"].median()
hr["experience"]=hr["experience"].fillna(exp1)
 


model = LinearRegression()
model1 = DecisionTreeClassifier()

X = hr.drop(columns=["salary"])
y =hr["salary"]

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.4,random_state=42)

model.fit(X_train,y_train)
#print(model.score(X_test,y_test))
#print(model.coef_)
#print(model.intercept_)


#prediction = model.predict([[10,7,8]])
#print(prediction)

new = {
    "experience":[1,2,3,4,5],
    "test_score":[8,8,6,10,9],
    "interview_score":[10,6,10,7,8] 
}

hr_new = pd.DataFrame(new)
#print(hr_new)

salary_new = model.predict(hr_new)
hr_new["New_Salary"] = salary_new

hr_new.to_csv("New_salary.csv")