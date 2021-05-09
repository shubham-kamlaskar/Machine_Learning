import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

petrol = pd.read_csv("petrol.csv")

#print(petrol)

model = LinearRegression()

X = petrol.drop(columns=["Sell_Price"])
y = petrol["Sell_Price"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model.fit(X_train,y_train)
#print(model.score(X_test,y_test))
#print(model.coef_)
#print(model.intercept_)
'''
plt.scatter(petrol["Year"],petrol["Sell_Price"])
plt.twinx()
plt.plot(petrol["Year"],petrol["Mileage"],"r")
plt.show()
'''

predict = model.predict([[20000,2]])
print(predict)