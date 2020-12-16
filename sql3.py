# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 06:23:24 2020

@author: SAAD
"""
import mysql.connector

from mysql.connector import Error

mydb = mysql.connector.connect(
host="localhost", user="root",password="",database="fyp" )
cursor = mydb.cursor()
student_id=77
s_name= "saad"
s_section="B"



sql = "UPDATE student SET student_name = %s,section= %s  WHERE student_id = %s"
cursor.execute(sql,(s_name,s_section,student_id))
mydb.commit()

print(cursor.rowcount, "record(s) affected")
