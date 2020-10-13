import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

# import data
df = pd.read_csv("../data/data.csv")

# understanding data
print("Shape:", df.shape)
print("\n---------------------------------\n")
print(df.head())
print("\n---------------------------------\n")
print(df.info())
print("\n---------------------------------\n")
print(df.describe())
print("\n---------------------------------\n")


# print additonal info
print(df["diagnosis"].value_counts())

# plots
ax, fig = plt.subplots()
ax = sns.heatmap(df.corr(), annot=True)
plt.show()

ax, fig = plt.subplots(figsize=(15, 20))
ax = df.hist(bins=10)
plt.show()