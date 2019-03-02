# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:02:08 2019

@author: webberl
"""

from bs4 import BeautifulSoup
import csv


#url = "https://www.spotrac.com/mlb/statistics/player/#name=&type=yearly&pctGames=50&season=2012%2C2018&conference=&team=&positions%5B%5D=74&positions%5B%5D=34&positions%5B%5D=61&positions%5B%5D=35&positions%5B%5D=36&positions%5B%5D=37&positions%5B%5D=38&positions%5B%5D=62&positions%5B%5D=41&positions%5B%5D=40&positions%5B%5D=39&positions%5B%5D=42&salary=545000%2C34083333&status%5B%5D=active&sort=17%2C1"

#raw_html = requests.get(url)

with open('stats.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    raw_html = open('stats.html').read()
    html = BeautifulSoup(raw_html, 'html.parser')
    
    table_headers = html.table.thead
    table_data = html.table.tbody
    
    col_headers = []
    for cell in table_headers.tr.select('th'):
        raw_text = str(cell.text)
        col_headers.append(raw_text)
    writer.writerow(col_headers)
    
    for row in table_data.select('tr'):
        row_data = []
        for cell in row.select('td'):
            raw_text = str(cell.text)
            row_data.append(raw_text)
        writer.writerow(row_data)
    
