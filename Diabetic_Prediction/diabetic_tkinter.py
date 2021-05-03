import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from tkinter import *

root = Tk()
root.title("Diabetic Prediction using ML")
root.geometry("290x250")
root.resizable(False,False)

def SubMit():
    Glucose = pd.read_csv("c:/users/shubham/desktop/sublime/ml/pandas/Glucose.csv")
    X = Glucose.drop(columns=["Outcome"])
    y = Glucose["Outcome"]
    model = DecisionTreeClassifier()
    model.fit(X,y)
    predicition = model.predict([[float(E2.get()),float(E3.get()),float(E4.get()),float(E5.get()),
                                float(E6.get()),float(E7.get()),float(E8.get()),float(E9.get())]]
                                )
    if predicition == [1]:
        print(L10.config(text="Diabetic"))
    if predicition == [0] :
        print(L10.config(text="Normal"))

def ReSet():
    E2.delete(0,END)
    E3.delete(0,END)
    E4.delete(0,END)
    E5.delete(0,END)
    E6.delete(0,END)
    E7.delete(0,END)
    E8.delete(0,END)
    E9.delete(0,END)
    L10.config(text="")
    


L1 =Label(root,text="Enter below details to get the results",width=40)
L1.grid(row=0,column=0,columnspan=2,sticky="NSWE",pady=2)
L2 = Label(root,text="Pregnancies",width=20)
L2.grid(row=1,column=0)
E2= Entry(root,width=20)
E2.grid(row=1,column=1)

L3 = Label(root,text="Glucose",width=20)
L3.grid(row=2,column=0)
E3= Entry(root,width=20)
E3.grid(row=2,column=1)

L4 = Label(root,text="Blood Pressure",width=20)
L4.grid(row=3,column=0)
E4= Entry(root,width=20)
E4.grid(row=3,column=1)

L5 = Label(root,text="Skin Thickness",width=20)
L5.grid(row=4,column=0)
E5= Entry(root,width=20)
E5.grid(row=4,column=1)

L6 = Label(root,text="Insulin",width=20)
L6.grid(row=5,column=0)
E6= Entry(root,width=20)
E6.grid(row=5,column=1)

L7 = Label(root,text="BMI",width=20)
L7.grid(row=6,column=0)
E7= Entry(root,width=20)
E7.grid(row=6,column=1)

L8 = Label(root,text="Diabetes Pedigree Function",width=20)
L8.grid(row=7,column=0)
E8= Entry(root,width=20)
E8.grid(row=7,column=1)

L9 = Label(root,text="Age",width=20)
L9.grid(row=8,column=0)
E9= Entry(root,width=20)
E9.grid(row=8,column=1)

B1 = Button(root,text='Submit',command=SubMit)
B1.grid(row=9,column=0,pady=2)
B2 = Button(root,text="Reset",command=ReSet)
B2.grid(row=9,column=1,pady=2)

L10 = Label(root,text="")
L10.grid(row=10,column=0,columnspan=2)


root.mainloop()

