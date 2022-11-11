import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import datetime 
import re

file_path = open(r"C:\Users\shubham\Desktop\Python\Kaggle\Carnivorous\carnivorous_diet.csv")

carni = pd.read_csv(file_path)
print(carni.head())
# print(carni.info())