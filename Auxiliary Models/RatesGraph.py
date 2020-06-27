# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 01:29:10 2020

@author: e-sshen
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('C:/Users/e-sshen/Desktop/GIC Data/RateCorrelation.csv')
df = pd.DataFrame(dataset)

week = df['Week']
gic_pur = df['GIC Purchase']
gic_rew = df['GIC Renewal']
l_gic_rate = df['Average Long GIC Rate']
s_gic_rate = df['Average Short GIC Rate']
csh_rate = df['Average CSH Rate']
sav_rate = df['SAV']
l_gic_rank = df['Average Long GIC Rank']
s_gic_rank = df['Average Short GIC Rank']
csh_rank = df['Average CSH Rank']
sav_rank = df['SAV Rank']

#plt.figure(1, figsize = (22, 8.5))
plt.plot(week, gic_pur, color='g', label = "GIC Purchases")
plt.xlabel('Week')
plt.plot(week, l_gic_rank, color = 'b', label = "Long GIC Rank")
plt.plot(week, s_gic_rank, color = 'y', label = "Short gic Rank")
plt.plot(week, csh_rank, color = 'r', label = "CSH Rank")
plt.plot(week, sav_rank, color = 'c', label = "SAV Rank")
plt.legend(loc="upper right")
plt.show()

plt.plot(week, gic_rew, color = 'g', label = "GIC Renewals")
plt.xlabel('Week')
plt.plot(week, l_gic_rank, color = 'b', label = "Long GIC Rank")
plt.plot(week, s_gic_rank, color = 'y', label = "Short gic Rank")
plt.plot(week, csh_rank, color = 'r', label = "CSH Rank")
plt.plot(week, sav_rank, color = 'c', label = "SAV Rank")
plt.legend(loc="upper right")
plt.show()

plt.plot(week, gic_rew, color = 'g', label = "GIC Renewals")
plt.plot(week, gic_pur, color='b', label = "GIC Purchases")
plt.ylabel('Number of Transactions')
plt.xlabel('Week')
plt.legend(loc="upper right")
plt.show()

