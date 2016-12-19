#!/usr/bin/env python
#-*- coding:utf -*-

import os,sys
import random

test_rate = 0.20

if __name__=='__main__':

  f = open(sys.argv[1],'rb')
  f1 = open("wx_marry.train","wb")
  f2 = open("wx_marry.test","wb")
  feas = f.readline()
  f1.write(feas)
  f2.write(feas)
  s = f.readline()
  while s:
    items = s.split(',')
    if (items[1] == '1' or items[1] == '2') and int(items[3]) >= 17 :
      rcnt = random.randint(1,100)	
      if rcnt < 100*test_rate:
        f2.write(s)	 
      else:
	f1.write(s)  
    s = f.readline()
