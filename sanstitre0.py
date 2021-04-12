# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 00:22:22 2020

@author: BS
"""
import os
for i in range (1,41):
    for j in range(1,4):

        if len(os.listdir('harris_orl_test\\'+str(i)+'\\'+str(j)) ) == 0:
            print("Directory is empty")
