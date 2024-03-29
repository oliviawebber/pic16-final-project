# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:23:46 2019

@author: webberl
"""

import pandas as pd
import numpy as np

df=pd.read_csv('updated_expanded_pitcher.csv', sep=',',header=0)

avg_salary = [371571,412520,412454,438729,497254,597537,851492,1028667,1076089,
              1168263,1110766,1119981,1336609,1398831,1611166,1895630,2138896,
              2295649,2372189,2313535,2476589,2699292,2824751,2925679,2996106,
              3014572,3095183,3210000,3390000,3690000,3840000,4380000,4470000,
              4520000]

inflation = (len(avg_salary)-1) * [0]
for i in range(len(avg_salary)-1):
    inflation[i] = float(avg_salary[i+1])/avg_salary[i]

years = range(1985,2017,1)


for i in range(len(years)):
    year = years[i]
    mask = df['yearID']==year
    for adjustment in inflation[i:]:
        df.loc[mask, 'salary'] = df[mask]['salary'] * (adjustment)
        
for i in range(len(years)):
    year = years[i]
    mask = df['yearID']==year-1
    for adjustment in inflation[i:]:
        df.loc[mask, 'prev_salary'] = df[mask]['prev_salary'] * (adjustment)

removed_outliers = (df.salary < np.percentile(df.salary, 75)*1.5) & (df.salary > np.percentile(df.salary, 25)/1.5)
df = df[removed_outliers]

#year_mask= df.yearID > 2010
#df = df[year_mask]

cutoff = 5
 
pitcher_mask = (df.W < cutoff) & \
       (df.L < cutoff) & \
       (df.G < cutoff) & \
       (df.GS < cutoff) & \
       (df.CG < cutoff) & \
       (df.SHO < cutoff) & \
       (df.SV < cutoff) & \
       (df.Ipouts < cutoff) & \
       (df.H < cutoff) & \
       (df.ER < cutoff) & \
       (df.HR < cutoff) & \
       (df.BB < cutoff) & \
       (df.SO < cutoff) & \
       (df.BAOpp < cutoff) & \
       (df.ERA < cutoff) & \
       (df.WP < cutoff) &\
       (df.HBP < cutoff) &\
       (df.BK < cutoff) &\
       (df.BFP < cutoff) &\
       (df.GF < cutoff) &\
       (df.R < cutoff) &\
       (df.SH < cutoff) &\
       (df.SF < cutoff) &\
       (df.GIDP < cutoff)
df = df.drop(df[mask].index)

df.to_csv("clean_pitcher.csv", index=False)
