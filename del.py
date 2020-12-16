# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:58:24 2020

@author: SAAD
"""
import mysql.connector
from datetime import datetime
import operator
import functools
from mysql.connector import Error



mydb = mysql.connector.connect(
host="localhost", user="root",password="",database="fyp" )
cursor = mydb.cursor()
   
sql = "DELETE FROM list"
cursor.execute(sql)
mydb.commit()
print(cursor.rowcount, "record(s) deleted")   
sql="""select * from courses"""
cour=cursor.execute(sql)
cour=cursor.fetchall()
for co in cour:
    print(co[1])