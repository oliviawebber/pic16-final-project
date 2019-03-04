# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:02:08 2019

@author: webberl
"""

from bs4 import BeautifulSoup
import csv


#with open('batter_stats.csv','w') as csvfile:
#    writer = csv.writer(csvfile)
#    raw_html = open('batter_stats.html').read()
#    html = BeautifulSoup(raw_html, 'html.parser')
#    
#    table_headers = html.table.thead
#    table_data = html.table.tbody
#    
#    col_headers = []
#    for cell in table_headers.tr.select('th'):
#        raw_text = str(cell.text)
#        col_headers.append(raw_text)
#    col_headers.insert(2, "Position")
#    writer.writerow(col_headers)
#    
#    for row in table_data.select('tr'):
#        row_data = []
#        for cell in row.select('td'):
#            raw_text = str(cell.text)
#            row_data.append(raw_text)
#        player_and_pos = row_data[1]
#        split = player_and_pos.rsplit('.',1)
#        row_data[1] = split[0]
#        row_data.insert(2, split[1][1:])
#        writer.writerow(row_data)

with open('batter_stats.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    raw_html = open('batter_stats.html').read()
    html = BeautifulSoup(raw_html, 'html.parser')
    
    table_headers = html.table.thead
    table_data = html.table.tbody
    
    col_headers = []
    for cell in table_headers.tr.select('th'):
        raw_text = str(cell.text)
        col_headers.append(raw_text)
    col_headers.insert(2, "Position")
    writer.writerow(col_headers)
    
    for row in table_data.select('tr'):
        row_data = []
        for i in range(len(row.select('td'))):
            cell = row.select('td')[i]
            raw_text = str(cell.text)
            
            if i < 4:
                row_data.append(raw_text)
            elif i >= 4 and i < 17:
                row_data.append(float(raw_text))
            
            else:
                raw_text = raw_text.replace('$', '')
                raw_text = raw_text.replace(',','')
                row_data.append(float(raw_text))
                
        player_and_pos = row_data[1]
        split = player_and_pos.rsplit('.',1)
        row_data[1] = split[0]
        row_data.insert(2, split[1][1:])
        writer.writerow(row_data)
    
with open('pitcher_stats.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    raw_html = open('pitcher_stats.html').read()
    html = BeautifulSoup(raw_html, 'html.parser')
    
    table_headers = html.table.thead
    table_data = html.table.tbody
    
    col_headers = []
    for cell in table_headers.tr.select('th'):
        raw_text = str(cell.text)
        col_headers.append(raw_text)
    col_headers.insert(2, "Position")
    writer.writerow(col_headers)
    
    for row in table_data.select('tr'):
        row_data = []
        for i in range(len(row.select('td'))):
            cell = row.select('td')[i]
            raw_text = str(cell.text)
            
            if i < 4:
                row_data.append(raw_text)
            elif i >= 4 and i < 17:
                row_data.append(float(raw_text))
            
            else:
                raw_text = raw_text.replace('$', '')
                raw_text = raw_text.replace(',','')
                row_data.append(float(raw_text))
                
        player_and_pos = row_data[1]
        split = player_and_pos.rsplit('.',1)
        row_data[1] = split[0]
        row_data.insert(2, split[1][1:])
        writer.writerow(row_data)