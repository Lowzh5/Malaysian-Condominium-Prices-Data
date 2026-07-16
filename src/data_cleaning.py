import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pd.set_option('display.max_columns', None)

csv_path = os.path.join(BASE_DIR, "data", "raw", "houses.csv")
df = pd.read_csv(csv_path)
print(df.head())

print(df.info())

print("-------------------------------")
print(df[df["Ad List"].isna() == True])

print("-------------------------------")
print(df["Facilities"].sample(10))

print("-------------------------------")
print(df.shape)

print(df.duplicated().sum())

#dasd