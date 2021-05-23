import pandas as pd

ship = pd.read_csv("Train.csv")
#print(ship)


import matplotlib.pyplot as plt
import seaborn as sns

ship = ship.drop(columns=["ID","Warehouse_block","Discount_offered"])
#print(ship.info)
#print(ship["Mode_of_Shipment"].unique())

#sns.countplot(ship["Gender"])
#sns.countplot(ship["Mode_of_Shipment"])
#sns.countplot(ship["Reached.on.Time_Y.N"])
#sns.countplot(ship["Customer_rating"])
#sns.countplot(ship["Customer_care_calls"])
#sns.countplot(ship["Product_importance"])
#sns.catplot(x="Weight_in_gms",y="Product_importance",data=ship,hue="Reached.on.Time_Y.N",kind="bar")
#sns.catplot(x="Weight_in_gms",y="Mode_of_Shipment",data=ship,hue="Reached.on.Time_Y.N",kind="bar")
#sns.catplot(x="Customer_rating",y="Cost_of_the_Product",data=ship,hue="Gender",kind="bar")
#sns.catplot(x="Product_importance",y="Cost_of_the_Product",data=ship,hue="Gender",kind="bar")
#sns.heatmap(ship.corr(),annot=True)
#sns.jointplot(x="Cost_of_the_Product",y="Weight_in_gms",data=ship,kind="hex")
#sns.jointplot(x="Weight_in_gms",y="Cost_of_the_Product",data=ship,kind="kde",fill=True)
#sns.clustermap(ship.corr(),annot=True,cmap='cividis')
plt.show()


mode = { 'Flight':0,"Ship":1,"Road":2}
ship["Mode_of_Shipment"] = ship["Mode_of_Shipment"].map(mode)

gender = {'M':0,"F":1}
ship["Gender"] = ship["Gender"].map(gender)

imp = { "low":0,"medium":1,"high":2}
ship["Product_importance"] = ship["Product_importance"].map(imp)
#print(ship)

from sklearn.model_selection import train_test_split

X = ship.drop(columns=["Reached.on.Time_Y.N"])
y = ship["Reached.on.Time_Y.N"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(X_train,y_train)
#print(model.score(X_test,y_test))
# score is 0.6438636363636364 for test size =0.4
# score is 0.6460606060606061 for test size =0.3

from sklearn.linear_model import LinearRegression
le = LinearRegression()

le.fit(X_train,y_train)
#print(le.score(X_test,y_test))
# score is 0.10665573910979675 for test size =0.4
# score is 0.10430786266302760 for test size =0.3

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)
#print(lr.score(X_test,y_test))
#score is 0.6400000000000000 for test size =0.4
#score is 0.6324242424242424 for test size =0.3

predictions = lr.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))