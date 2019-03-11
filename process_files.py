#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:29 2019

@author: layton
"""
import csv
import pandas as pd

batting_data=pd.read_csv('Batting.csv', sep=',',header=0)
salary_data = pd.read_csv('Salaries.csv', sep=',',header=0)

year_mask = batting_data['yearID'] > 1984
batting_data = batting_data[year_mask]

with open('test_bat.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for row in batting_data.itertuples():
        playerID = getattr(row, 'playerID')
        yearID = getattr(row, 'yearID')
        teamID = getattr(row, 'teamID')
        player_mask = salary_data['playerID'] == playerID
        year_mask = salary_data['yearID'] == yearID
        team_mask = salary_data['teamID'] == teamID
        
        try:
            salary = salary_data[player_mask & year_mask & team_mask]['salary'].values[0,]
        except:
            salary = -1
            
        if salary != -1:
            player = list(row)
            player.append(salary)
            writer.writerow(player)
        