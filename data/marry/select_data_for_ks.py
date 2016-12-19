#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os,sys
import csv
import random

feas = ["uin","datasource","flag","ds","marry_satus"]
csv_file = "good_bad_with_beizhu.csv"
test_rate = 0.20
for row in csv.DictReader(open(csv_file)):
  fdate = "201611"
  if random.randint(0,1000) > 1000*test_rate:
      fdate = "201610"
  li = [row['useruin'],"train","A,0,0,0,0,0,0,0,0,0,0,0,0="+str(row['credict']),
        fdate,row["fage"]]
  print "\t".join(li)
 
