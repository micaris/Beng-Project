import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew, pearsonr
import seaborn as sns

df = pd.read_csv('datasets/bank.csv')
print(df.head())
