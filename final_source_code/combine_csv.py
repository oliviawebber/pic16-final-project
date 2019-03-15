#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:29 2019

@author: layton
"""
import csv
import pandas as pd
import re

datasets = ['Batting.csv', 'Pitching.csv']

for dataset in datasets:
    stat_data=pd.read_csv(dataset, sep=',',header=0)
    salary_data = pd.read_csv('Salaries.csv', sep=',',header=0)
    
    year_mask = stat_data['yearID'] > 1984
    stat_data = stat_data[year_mask]
    
    
    file_name = 'salary_and_stat_' + dataset
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        
        headers = list(stat_data.columns.values)
        headers.append('prev_salary')
        headers.append('salary')
        headers.insert(0,'num')
        regex = re.compile(r'^(\d*)')
        
        for i in range(len(headers)):
            entry = headers[i]
            match = regex.match(entry)
            digits = match.group(1)
            if digits != '':
                entry = regex.sub('', entry)
                entry += digits
            headers[i] = entry
                
        writer.writerow(headers)
        for row in stat_data.itertuples():
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
        