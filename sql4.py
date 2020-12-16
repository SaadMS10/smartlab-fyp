# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 03:58:52 2020

@author: SAAD
"""
import mysql.connector
from datetime import datetime
from mysql.connector import Error
try:
    mydb = mysql.connector.connect(
        host="localhost", user="root",password="",database="fyp" )
    cursor = mydb.cursor()
    sql = "DELETE FROM list"
    cursor.execute(sql)
    mydb.commit()
    print(cursor.rowcount, "record(s) deleted")   
    import operator
    import functools
    sql = """select * from student"""
    cursor.execute(sql)
    res2=cursor.fetchall()
    print(res2)
    sql="""select * from courses"""
    cour=cursor.execute(sql)
    cour=cursor.fetchall()
    for co in cour:


        for s in res2:
            sql = """select stu_name from list where (cour_id = %s && stu_name= %s)"""
            res= cursor.execute(sql, (co[0],s[0],))
            res=cursor.fetchall()
            if res== []:
                sql = """select count(stu_name)from attendance where (cour_id = %s && stu_name= %s)"""
                res= cursor.execute(sql, (co[0],s[0],))
                res=cursor.fetchone()
                result = int(''.join(map(str, res)))  
                print(result)
                sql = """select  count(DISTINCT cour_id,week)from attendance where cour_id= %s"""
                res1= cursor.execute(sql, (co[0],))
                res1=cursor.fetchone()
                result1 = int(''.join(map(str, res1)))  
                if (result == 0 | result1 == 0):  
                    star=0
                    sql = """INSERT INTO list (stu_name,cour_id,course_name,percent)  VALUES (%s,%s,%s,%s)"""
                    val=(s[1],co[0],co[1],star)
                    cursor.execute(sql,val)
                    mydb.commit()
                    print("commited")
                    print("Attendance Percentange: " + str(star) + " %")
                    sql = """select * from list"""
                    cursor.execute(sql)
                    attendance=cursor.fetchall()
                else:
                    percent = tuple(map(operator.truediv, res, res1))   
                    star = (functools.reduce(operator.add, (percent)))*100
                    sql = """INSERT INTO list (stu_name,cour_id,course_name,percent)  VALUES (%s,%s,%s,%s)"""
                    val=(s[1],co[0],co[1],star)
                    cursor.execute(sql,val)
                    mydb.commit()
                    print("commited")
                    print("Attendance Percentange: " + str(star) + " %")
                    sql = """select * from list"""
                    cursor.execute(sql)
                    attendance=cursor.fetchall()
except Error as e:
    print("Error reading data from MySQL table", e)
    