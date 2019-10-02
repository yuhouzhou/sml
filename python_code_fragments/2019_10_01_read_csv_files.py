#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Simple example of how to read in CSV files
"""

import csv

reader = csv.reader(open('test.csv', 'r'))
data = [row for row in reader] # store every row in a list called data

print(data)

for row in data:
    print(row)
    
for row in data:
    if row[0] != 'Student':
        print(row[0],'achieved the grade',row[1])