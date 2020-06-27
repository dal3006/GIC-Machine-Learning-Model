# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:50:36 2020

@author: e-sshen
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('C:/Users/e-sshen/Desktop/GIC Data/CorrelationData.csv')
df = pd.DataFrame(dataset)
columns = ['Index', 'Holder', 'Average CSH Rank']
df.drop(columns, inplace = True, axis = 1)

def CorrMtx(df, dropDuplicates = True):
    df = df.corr()
    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:    
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sns.set_style(style = 'white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)

CorrMtx(df, dropDuplicates = False)