# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:06:33 2020

@author: SAAD
"""

from flask import Flask,redirect
from flask_mail import Mail, Message
import mysql.connector



app = Flask(__name__)
mail= Mail(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'labsmrt'
app.config['MAIL_PASSWORD'] = 'finalyear'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

@app.route("/")
def mail():
   mydb = mysql.connector.connect(
    host="localhost", user="root",password="",database="fyp" )
   cursor = mydb.cursor()
   sql="select * from list"
   cursor.execute(sql)
   res=cursor.fetchall()
   for r in res:
       if float(r[3]) < 75.0:
            sql = """select * from student where student_name= %s"""
            cursor.execute(sql,(r[0],))
            ml=cursor.fetchall()
            for m in ml:
                
                msg = Message('Short Attendance', sender = 'labsmrt', recipients = [m[3]])
                msg.body = "Hello " + r[0] + "  You're In The Defaulter List Of Course " + r[2] + " Your Attendance Is " +r[3] + "%"
                mail.send(msg)
   return redirect ("http://mail.google.com")
                
          
       


if __name__ == '__main__':
   app.run(debug = True)