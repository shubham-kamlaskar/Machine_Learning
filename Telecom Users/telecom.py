import numpy as np
import pandas as pd

tel = pd.read_csv("telecom_users.csv")
#print(tel.head(10))


tel['TotalCharges'] = pd.to_numeric(tel['TotalCharges'],errors = 'coerce')
tel.TotalCharges = tel.TotalCharges.fillna(tel.TotalCharges.mean())
#print(tel.TotalCharges.isnull().sum())
#print(tel.info())
#print(tel.describe())
#print(tel.isnull().sum())
'''
print(tel.gender.unique())
print(tel.Partner.unique())
print(tel.Dependents.unique())
print(tel.PhoneService.unique())
print(tel.MultipleLines.unique())
print(tel.InternetService.unique())
print(tel.OnlineSecurity.unique())
print(tel.OnlineBackup.unique())
print(tel.DeviceProtection.unique())
print(tel.TechSupport.unique())
print(tel.StreamingTV.unique())
print(tel.StreamingMovies.unique())
print(tel.Contract.unique())
print(tel.PaperlessBilling.unique())
print(tel.PaymentMethod.unique())
print(tel.Churn.unique())
'''

tel = tel.drop(columns=["No","customerID"])

import matplotlib.pyplot as plt 
import seaborn as sns

#sns.heatmap(tel.corr(),annot=True)
#sns.countplot(tel.gender)
#sns.countplot(tel.SeniorCitizen)
#sns.countplot(tel.Partner)
#sns.countplot(tel.Dependents)
#sns.countplot(tel.Contract)
#sns.countplot(tel.PaperlessBilling)
#sns.countplot(tel.PaymentMethod)
#sns.countplot(tel.Churn)
#sns.countplot(tel.InternetService)
#sns.pairplot(tel)
#sns.boxplot(x="gender",y="TotalCharges",data=tel)
#sns.countplot(x="gender",hue="Churn",data=tel)
#sns.countplot(x="Partner",hue="Churn",data=tel)
#sns.countplot(x="Dependents",hue="Churn",data=tel)
#sns.countplot(x="InternetService",hue="Churn",data=tel)
#plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

tel.gender = le.fit_transform(tel.gender)
tel.Partner = le.fit_transform(tel.Partner)
tel.Dependents = le.fit_transform(tel.Dependents)
tel.PhoneService = le.fit_transform(tel.PhoneService)

tel.MultipleLines = le.fit_transform(tel.MultipleLines)

tel.InternetService = le.fit_transform(tel.InternetService)
tel.OnlineSecurity = le.fit_transform(tel.OnlineSecurity)
tel.OnlineBackup = le.fit_transform(tel.OnlineBackup)
tel.DeviceProtection = le.fit_transform(tel.DeviceProtection)
tel.TechSupport = le.fit_transform(tel.TechSupport)
tel.StreamingTV = le.fit_transform(tel.StreamingTV)
tel.StreamingMovies = le.fit_transform(tel.StreamingMovies)
tel.Contract = le.fit_transform(tel.Contract)
tel.PaperlessBilling = le.fit_transform(tel.PaperlessBilling)
tel.PaymentMethod = le.fit_transform(tel.PaymentMethod)
tel.Churn = le.fit_transform(tel.Churn)

#sns.heatmap(tel.corr(),annot=True)
#sns.distplot(tel['MonthlyCharges'])
#sns.distplot(tel["TotalCharges"])
#sns.jointplot("tenure","MonthlyCharges",data=tel,kind="reg")
#sns.pairplot(tel,hue = 'Churn')
plt.show()

from sklearn.model_selection import train_test_split

X = tel.drop(columns=["Churn"]) 

y = tel.Churn


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42)

from sklearn.tree import DecisionTreeClassifier
de = DecisionTreeClassifier()

de.fit(X_train,y_train)
#print(de.score(X_test,y_test))
# score of 0.7360801781737194 by test size = 0.3
# score of 0.7360801781737194 by test size = 0.4

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train)

#print(lr.score(X_test,y_test))
# score of 0.24782725096096858 by test size = 0.3
# score of 0.25975618885970030 by test size = 0.4

from sklearn.linear_model import LogisticRegression
lre = LogisticRegression()

lre.fit(X_train,y_train)

#print(lre.score(X_test,y_test))
# score of 0.7989977728285078 by test size = 0.3
# score of 0.7903966597077244 by test size = 0.4

from sklearn.metrics import confusion_matrix
y_pred = lre.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
#print('confusion matrix:\n',cm)

from sklearn.metrics import accuracy_score
lra = accuracy_score(y_test,y_pred)
#print('accuracy score = ',lra)
# score of 0.7989977728285078 by test size = 0.3
# score of 0.7903966597077244 by test size = 0.4



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
#print(rfc.score(X_test,y_test))
# score of 0.7783964365256125 by test size = 0.3
# score of 0.7828810020876826 by test size = 0.4

from sklearn.metrics import confusion_matrix
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
#print('confusion matrix:\n',cm)

from sklearn.metrics import accuracy_score
lra = accuracy_score(y_test,y_pred)
#print('accuracy score = ',lra)
# score of 0.787305122494432 by test size = 0.3
# score of 0.7795407098121085 by test size = 0.4

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
y_pred = knc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
#print('confusion matrix:\n',cm)

from sklearn.metrics import accuracy_score
lra = accuracy_score(y_test,y_pred)
#print('accuracy score = ',lra)
# score of 0.7603340292275574 by test size = 0.3
# score of 0.7903966597077244 by test size = 0.4

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
y_pred = svc.predict(X_test)
#print('confusion matrix:\n',cm)

from sklearn.metrics import accuracy_score
lra = accuracy_score(y_test,y_pred)
#print('accurancy score =' ,lra)
# score of 0.7422048997772829 by test size = 0.3
# score of 0.7398747390396659 by test size = 0.4

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)

#getting confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = nb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('confusion matrix:\n',cm)

#checking accuracy
from sklearn.metrics import accuracy_score
nba = accuracy_score(y_test,y_pred)
print('accuracy score = ',accuracy_score(y_test,y_pred))
# score of 0.7488864142538976 by test size = 0.3
# score of 0.7507306889352818 by test size = 0.4


#conclusion is that , We got the maximum accuracy score of 0.7989977728285078 by LogisticRegression



