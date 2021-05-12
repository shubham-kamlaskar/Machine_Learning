from matplotlib.markers import MarkerStyle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import datetime

Nashik = pd.read_csv("Nashik.csv")


Nashik = Nashik.drop(columns=["Other","Tested","State","District"])
#print(Nashik.tail(29))


Nashik['Date'] = pd.to_datetime(Nashik['Date'])
Nashik['Date'] = Nashik['Date'].dt.strftime('%d.%m.%Y')
Nashik['year'] = pd.DatetimeIndex(Nashik['Date']).year
#Nashik['month'] = pd.DatetimeIndex(Nashik['Date']).month
#Nashik['day'] = pd.DatetimeIndex(Nashik['Date']).day
Nashik['dayofyear'] = pd.DatetimeIndex(Nashik['Date']).dayofyear
#Nashik['weekofyear'] = pd.DatetimeIndex(Nashik['Date']).weekofyear
#Nashik['weekday'] = pd.DatetimeIndex(Nashik['Date']).weekday
#Nashik['quarter'] = pd.DatetimeIndex(Nashik['Date']).quarter
#Nashik['is_month_start'] = pd.DatetimeIndex(Nashik['Date']).is_month_start
#Nashik['is_month_end'] = pd.DatetimeIndex(Nashik['Date']).is_month_end


Nashik1 = Nashik.tail(29)

Nashik1 = Nashik1.drop(columns=["Date"])

Nashik1 = pd.get_dummies(Nashik1, columns=['year'], drop_first=True, prefix='year')


model = LinearRegression()

y = Nashik1.drop(columns=["dayofyear"])
X = Nashik1["dayofyear"].values.reshape(-1,1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model.fit(X_train,y_train)

#print(model.score(X_test,y_test))

fut_dates = {
    "dayofyear":[132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160]
}

Date = pd.DataFrame(fut_dates)


prediction = model.predict(Date)
pred = pd.DataFrame(prediction,columns=["Confirmed","Recovered","Deceased"])
#
New = pd.concat([pred,Date],axis=1)


All = pd.concat([Nashik1,New])
All["Active"] = All["Confirmed"] - All["Recovered"] - All["Deceased"]
All = All[["dayofyear","Confirmed","Recovered","Deceased","Active"]]
#print(All)

#All.to_csv("predicted.csv",index=False)
'''
#visualization through graphs

#sns.jointplot(x="dayofyear",y="Active",data=All,kind="reg")
#sns.pairplot(data=All)
plt.figure("Corona Cases")
plt.subplot(2,2,1)
plt.ylabel("Active")
plt.plot(All["dayofyear"],All["Active"])
plt.subplot(2,2,2)
plt.xlabel("dayofyear")
plt.ylabel("Deceased")
plt.plot(All["dayofyear"],All["Deceased"],'r')
plt.subplot(2,2,3)
plt.ylabel("Confirmed")
plt.plot(All["dayofyear"],All["Confirmed"],'g')
plt.subplot(2,2,4)
plt.xlabel("dayofyear")
plt.ylabel("Recovered")
plt.plot(All["dayofyear"],All["Recovered"],'y')

plt.figure("Total Cases")
plt.xlabel("dayofyear")
plt.ylabel("No of Cases")
plt.bar(All["dayofyear"],All["Active"],color="b")

plt.bar(All["dayofyear"],All["Deceased"],bottom=All["Active"],color="r")
plt.bar(All["dayofyear"],All["Recovered"],bottom=All["Active"]+All["Deceased"],color="g")
plt.legend(["Active","Deceased","Recovered"])

plt.figure("Stackplot")
plt.stackplot(All["dayofyear"],All["Recovered"],All["Active"],All["Deceased"])
plt.legend(["Recovered","Active","Deceased"])

plt.show()
'''