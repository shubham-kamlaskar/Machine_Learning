import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


Home ={
    "Area":[2600,3000,3200,3600,4000],
    "Price":[550000,565000,610000,680000,725000]
}

Rate = pd.DataFrame(Home)
#print(Rate)

#sns.heatmap(Rate.corr())
#sns.boxplot(x="Area",y="Price",data=Rate)
#sns.jointplot(x="Area",y="Price",data=Rate,kind="hex")
#sns.pairplot(Rate)
#sns.displot(Rate)
#plt.show()
model = DecisionTreeClassifier()

X = Rate.drop(columns=["Price"])
y = Rate["Price"]

X_train,X_test,y_train,y_test =train_test_split(X,y,train_size=0.5,random_state=1)

model =LinearRegression()
model.fit(X_train,y_train)
print(model.score(X_test, y_test))
prediction = model.predict([[10000]])
#print("Total price is ",prediction)
#print(model.intercept_)
#print(model.coef_)

y =130 *3400 + 212000
#print(y)



df1 = {
    "Area":[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
}

df =pd.DataFrame(df1)
#print(df)

New_price  = (model.predict(df))
df["Prices"] =New_price
#print(df)

#df.to_csv("Prediction.csv",index=False)

plt.xlabel("Area in Sq.ft")
plt.ylabel("Price in Rs.")
plt.scatter(Rate["Area"],Rate["Price"],marker="o")
plt.plot(df["Area"][1:4],df["Prices"][1:4],'r')
plt.grid()
plt.show()