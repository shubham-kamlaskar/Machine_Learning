import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

Startup = pd.read_csv("50_Startups.csv")
#print(Startup)

#sns.heatmap(Startup.corr())
#sns.jointplot(x=Startup["State"],y=Startup["R&D Spend"])
#plt.bar(Startup["State"], Startup["R&D Spend"])

#plt.xlabel("State")
#plt.ylabel("Profit")

plt.bar(Startup["State"],Startup["Profit"])


#plt.show()

#print(Startup.info())
#print(Startup.values)
print(Startup.sum())

a= Startup.groupby(["State"]).mean()
print(a)
a.to_csv("State wise stats.csv")

'''
Startup2 = Startup.drop(columns=["State"])
#print(Startup2)

Startup1 = pd.get_dummies(Startup["State"])
#print(Startup1)



frames = [Startup1,Startup2]

Startup = pd.concat(frames,axis=1)
#print(Startup)

model = LinearRegression()

X = Startup.drop(columns=["Profit"])
y = Startup["Profit"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model.fit(X_train,y_train)

#print(model.score(X_test,y_test))

predict = model.predict([[1,0,0,130256.7,91814.77,345678]])
#print("Profit is Rs. ",predict)

#joblib.dump(model,"50_startup")


sns.pairplot(Startup,hue="State")
plt.show()
'''