#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:29 2019

@author: layton
"""
import csv
import pandas as pd

batting_data=pd.read_csv('Pitching.csv', sep=',',header=0)
salary_data = pd.read_csv('Salaries.csv', sep=',',header=0)

year_mask = batting_data['yearID'] > 1984
batting_data = batting_data[year_mask]

with open('updated_expanded_pitcher.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for row in batting_data.itertuples():
        playerID = getattr(row, 'playerID')
        yearID = getattr(row, 'yearID')
        teamID = getattr(row, 'teamID')
        player_mask = salary_data['playerID'] == playerID
        cur_year_mask = salary_data['yearID'] == yearID
        next_year_mask = salary_data['yearID'] == yearID + 1
        team_mask = salary_data['teamID'] == teamID
        
        try:
            salary = salary_data[player_mask & cur_year_mask & team_mask]['salary'].values[0,]
            next_salary = salary_data[player_mask & next_year_mask & team_mask]['salary'].values[0,]
        except:
            salary = -1
            next_salary = -1
            
        
        if salary != -1 and next_salary != -1:
            player = list(row)
            player.append(salary)
            player.append(next_salary)
            writer.writerow(player)
        