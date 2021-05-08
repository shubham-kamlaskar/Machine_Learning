import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Capita = pd.read_csv("c:/users/shubham/desktop/sublime/ml/sci-learn/per capita income/canada_per_capita_income.csv")

model = DecisionTreeClassifier()
X = Capita.drop(columns=["income"])
y = Capita["income"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
model = LinearRegression()
model.fit(X_train,y_train)
#print(model.score(X_test,y_test))
#print(model.intercept_)
#print(model.coef_)
future = model.predict([[2017]])
#print(future)

F = {
    "Year":[2017,2018,2019,2020,2021,2022,2023,2024,2025]
}

Future = pd.DataFrame(F)
#print(Future)

P = model.predict(Future)
print(P)

Future["Future Income"] = P
print(Future)

plt.scatter(X,y)
plt.plot(Future["Year"],Future["Future Income"])
sns.jointplot(x=Capita["year"],y=Capita["income"],kind="reg")
plt.show()