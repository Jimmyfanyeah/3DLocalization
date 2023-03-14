# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:18:15 2021

@author: mastaffs
"""
import difflib
import os

filename = 'calc_metrics.py'

path1 = r'C:\Users\DLJ\Desktop\upsample2\utils'
path2 = r'C:\Users\DLJ\Desktop\forward\utils'

filenames = os.listdir(path1)
for filename in filenames:
    file1 = open(os.path.join(path1,filename),'r',encoding='utf-8').readlines()
    try:
        file2 = open(os.path.join(path2,filename),'r',encoding='utf-8').readlines()
        d=difflib.HtmlDiff()
        results=d.make_file(file1,file2) # 返回HTML形式的比较字符串
        with open(os.path.join(path1,filename.split('.')[0]+'.html'),'w') as file:
        	file.write(results)	# 将比较结果保存在results.html文件中
        print(filename)
    except: pass

filenames = filenames[2:-1]