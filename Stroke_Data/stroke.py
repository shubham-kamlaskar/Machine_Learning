import pandas as pd

stroke = pd.read_csv("stroke_data.csv")

stroke["bmi"] = stroke["bmi"].fillna(stroke["bmi"].mean())

stroke = stroke.drop(columns=["id"])

#marr = {"Yes":1,"No":0}
#troke["ever_married"] = stroke["ever_married"].map(marr)

#dummies = pd.get_dummies(stroke[["ever_married","work_type"]])

import matplotlib.pyplot as plt
import seaborn as sns

#sns.heatmap(stroke.corr())
#sns.countplot(stroke["work_type"])
#sns.countplot(stroke["Residence_type"])
#sns.countplot(stroke["smoking_status"])
#sns.pairplot(stroke)
#plt.title("age vs bmi")
#plt.xlabel("age")
#plt.ylabel("bmi")
#plt.scatter(stroke["age"],stroke["bmi"])
#sns.jointplot(stroke["age"],stroke["bmi"],kind="reg")
#sns.countplot(stroke["stroke"])
#sns.countplot(stroke["gender"])
#sns.countplot(stroke["ever_married"])
#plt.show()

min_glucose = min(stroke["avg_glucose_level"])
max_glucose = max(stroke["avg_glucose_level"])
min_age = min(stroke["age"])
max_age = max(stroke['age'])
min_bmi = min(stroke["bmi"])
max_bmi = max(stroke["bmi"])

#print(min_glucose,max_glucose,min_age,max_age,min_bmi,max_bmi)

#sns.displot(stroke["age"])
#sns.displot(stroke["avg_glucose_level"])
#plt.show()

work = {"Private":0,"Self-employed":1,"Govt_job":2,"children":3,"Never_worked":4}
stroke["work_type"] = stroke["work_type"].map(work)

gender = {"Male":0,"Female":1,"Other":3}
stroke["gender"] = stroke["gender"].map(gender)

married = {"Yes":0,"No":1}
stroke["ever_married"] = stroke["ever_married"].map(married)

Residence = {"Urban":0,"Rural":1}
stroke["Residence_type"] = stroke["Residence_type"].map(Residence)

smoking = {"formerly smoked":0,"never smoked":1,"smokes":2,"Unknown":3}
stroke["smoking_status"] = stroke["smoking_status"].map(smoking)

#print(stroke.head(10))

#sns.heatmap(stroke.corr(),annot=True)
#sns.scatterplot(stroke["age"],stroke["avg_glucose_level"])
#sns.catplot(x='heart_disease',y='age', hue="work_type", kind="bar", data=stroke)
#sns.catplot(x='heart_disease',y='age', hue="smoking_status", kind="bar", data=stroke)
#sns.catplot(x='heart_disease',y='age', hue="ever_married", kind="bar", data=stroke)
#sns.catplot(x='heart_disease',y='age', hue="hypertension", kind="bar", data=stroke)
#sns.catplot(x='stroke', y="avg_glucose_level", kind="box", data=stroke)
plt.show()


from sklearn.model_selection import train_test_split
X =stroke.drop(columns=["stroke"])

y =stroke["stroke"]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


#print(model.score(X_test,y_test))

#LOGISTIC REGRESSION IS PERFORMING WELL, BUT CAN WE IMPROVE PERFORMANCE USING ANOTHER MODEL? LET'S APPLY ANOTHER ALGORITHM

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print(rfc.score(X_test,y_test))

predict = rfc.predict([[0,81,0,1,0,1,0,71,45,0]])
print(predict)
