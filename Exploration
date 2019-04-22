# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# Data Pre-processing

train_df = pd.read_csv("train.csv", parse_dates=["first_active_month"]) #makes first_active_month column as datetime column 
test_df = pd.read_csv("test.csv", parse_dates=["first_active_month"])

print("Number of rows and columns in train set : ",train_df.shape)
print("Number of rows and columns in test set : ",test_df.shape)
