from matplotlib.markers import MarkerStyle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

all = pd.read_csv("districts.csv")
#print(all.head(10))

mh = all[all.State=="Maharashtra"]

#mh["Tested"] = mh["Tested"].fillna(0)

nsk = mh[mh.District=="Nashik"]
#nsk.to_csv("Nashik.csv")


nsk1= nsk.tail(30)

nsk1 = nsk.drop(columns=["Other","Tested","State","District"])
#print(nsk1)



#nsk1['Date'] = nsk1['Date'].str.replace('-','').astype(float)
#nsk1[["Year","Month","Date"]] = nsk1["Date"].str.split("-", expand = True)
nsk1["Date"]=pd.to_datetime(nsk1["Date"])
nsk1["Year"]= nsk1["Date"].dt.year
nsk1["Month"]=nsk1["Date"].dt.month
nsk1["Day"]=nsk1["Date"].dt.day

''''
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")

plt.plot(nsk1["Date"][:-29],nsk1["Confirmed"][:-29],label="Date Vs Confirmed")
plt.xticks(rotation=40)
plt.legend(loc="upper left")

plt.twinx()
plt.ylabel("Recovered Cases")
plt.plot(nsk1["Date"][:-29],nsk1["Recovered"][:-29],"-r",label="Date vs Recovered")
plt.xticks(rotation=40)
plt.legend(loc="upper right")
plt.grid()
#plt.bar(nsk1["Date"][:-29],nsk1["Deceased"][:-29])
plt.show()

nsk2 = nsk1.tail(29)
sns.heatmap(nsk2.corr())
plt.show

'''
nsk1["Yr-Mon"] = pd.concat(nsk1.Year,nsk1.Month)
print(nsk1["Yr-Mon"])
'''
nsk2 = nsk1.tail(29)
month = {4:0,5:1}
nsk2["Month"] = nsk2["Month"].map(month)

le = LabelEncoder()
nsk2["Day"] = le.fit_transform(nsk2["Day"])

#print(nsk2)

model = LinearRegression()

X = nsk1.drop(columns=["Day"])
y = nsk1["Day"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
pred = model.predict([[]])
'''