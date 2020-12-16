from flask import Flask, render_template, request, session, redirect,url_for
import json
import mysql.connector

from mysql.connector import Error
import os
from datetime import datetime
import face_recognition
import pickle
import cv2
import pyttsx3 
global week
global courses
week=0
incharge=[]



with open('config.json', 'r') as c:
    params = json.load(c)["params"]

local_server = True
app = Flask(__name__)

app.secret_key='saad'

mydb = mysql.connector.connect(
host="localhost", user="root",password="",database="fyp" )
cursor = mydb.cursor()

@app.route("/") #saad
def main():
    return render_template('main.html', params=params)
@app.route("/admin_login", methods = ['GET', 'POST'])# saad
def admin_login():
    
   
    if('user' in session and session['user']== params['admin_user']):
        return redirect('/dashboard')
    
    if request.method=="POST":
        username= request.form.get('username')
        password= request.form.get('password')
        if(username == params['admin_user'] and password == params['admin_pass']):
            #set the session variable
            session['user'] = username
            return redirect('/dashboard')
    
    return render_template('admin_login.html', params=params)
@app.route("/login", methods = ['GET', 'POST'])# saad
def login():
    global inst
    inst= ' '
    if('user' in session and session['user']== inst):
        return redirect('/home')
    if request.method=="POST":
        username1= request.form.get('username')
        password1= request.form.get('password')
        inst=username1
        print(username1)
        print(inst)
        sql="select * from instructor"
        cursor.execute(sql)
        instructor=cursor.fetchall()
        for x in instructor:      
            if(username1 == x[4] and password1 == x[2]):
                session['user'] = username1
                global usr
                usr=x[1]
                return redirect('/home')
    if('user' in session and session['user']== inst):
        return redirect('/home')
    return render_template('login.html')
    
@app.route("/dashboard")# saad
def dashboard():
        if('user' in session and session['user']== params['admin_user']):
            usr= session['user']
            mydb = mysql.connector.connect(
            host="localhost", user="root",password="",database="fyp" )
            cursor = mydb.cursor()
            sql = "select * from student"
            student= cursor.execute(sql) 
            
            
            student=cursor.fetchall()
            sql="select COUNT(*) from student"
            cursor.execute(sql)
            s_c= cursor.fetchone()
            print(student)
            return render_template('dashboard.html',student=student,usr=usr,s_c=s_c)
        else:
            return render_template('admin_login.html')
@app.route("/instructor")# saad
def instructor():
        if('user' in session and session['user']== params['admin_user']):
            usr= session['user']
            mydb = mysql.connector.connect(
            host="localhost", user="root",password="",database="fyp" )
            cursor = mydb.cursor()
            sql = "select * from instructor"
            intructor= cursor.execute(sql) 
            instructor=cursor.fetchall()
            sql="select COUNT(*) from instructor"
            cursor.execute(sql)
            i_c= cursor.fetchone()
            sql="select COUNT(*) from student"
            cursor.execute(sql)
            s_c= cursor.fetchone()
            print(instructor)
            return render_template('instructor.html',instructor=instructor,usr=usr,s_c=s_c,i_c=i_c)
        else:
            return render_template('admin_login')
@app.route("/home", methods = ['GET', 'POST']) #saad
def home():
        if('user' in session and session['user']== inst):
           return render_template('home.html',inst=inst,usr=usr)
       
        return render_template('login.html',inst=inst)
@app.route("/model", methods = ['GET', 'POST'])# saad
def model():
    if(request.method=='POST'):
        global courses
        week = 0

        courses = request.form.get('courses')
        week = request.form.get('week')
        engine = pyttsx3.init() 
        print(week)
        def relay(a):
            print("PANEL "+ a)
        cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
        faceCascade = cv2.CascadeClassifier(cascPathface)
        data = pickle.loads(open('recognizer', "rb").read())
        def mark_attendance(n):     
            try:
                now = datetime.now()
                nameList=[]
                sql = """select course_name from courses where course_code = %s"""
                cursor.execute(sql, (courses,))
                myresult = cursor.fetchone()
                #print(myresult)
                for line in myresult:
                    entry = line.split(',')
                    nameList.append(entry[0])
                    #print(nameList[0])
                    now= datetime.now()
                dtString= now.strftime('%I_%M_%S_%p')
                print(dtString)
                sql = """select student_id from student where student_id= %s"""
                user= cursor.execute(sql, (name,))
                res=cursor.fetchall()
                if(res == []):
                    sql = """select  teacher_id from instructor where teacher_id = %s"""
                    user= cursor.execute(sql, (name,))
                    res=cursor.fetchall()   
                    if res != []:
                        if name not  in incharge:
                            print("----------------------------")
                            print("Lab Incharge " +name+ " Arrived")
                            engine.say("Lab Incharge " +name+ " Arrived") 
                            relay("ON")
                            incharge.append(name)

                    else:
                        ("NOTHING")
                else:        
                    sql = """select stu_name from attendance where (stu_name= %s && week = %s)"""
                    user= cursor.execute(sql, (name,week))
                    res1=cursor.fetchall()
                    print(res1)
                    if res1 ==[]:
                        sql = """INSERT INTO attendance (stu_name,cour_id,week,datetime)  VALUES (%s,%s,%s,%s)"""
                        val=(name,courses,week,dtString)
                        cursor.execute(sql,val)
                        mydb.commit()
                        print("Attendance Marked")
                        engine.say("Attendance Marked ThankYou") 
                        print(cursor.rowcount, "Attendance Marked.") 
                    else:
                        print("Attendance Already Marked")
                        engine.say("Attendance Already Marked")
            except Error as e:
                
                print("Error reading data from MySQL table", e)
        print("Streaming started")
        video_capture = cv2.VideoCapture(1)
        while True:
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            names = []
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"],encoding)
                name = "Unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)
                for ((x, y, w, h), name) in zip(faces, names):

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
                    engine.runAndWait()
                    mark_attendance(name)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video_capture.release()
    cv2.destroyAllWindows()
    #return redirect('/attendance')
    return redirect(url_for("attendance",wek=week))
@app.route("/editstudent/<string:student_id>", methods =['GET','POST']) #saad
def editstudent(student_id):
    if('user' in session and session['user']== params['admin_user']):
        if request.method == 'POST':
            s_id= request.form.get('student_id')
            s_name= request.form.get('student_name')
            s_cnic= request.form.get('cnic')
            s_email= request.form.get('email')
            s_no= request.form.get('student_no')
            s_department= request.form.get('department')
            s_section=request.form.get("section")
            print(s_name)
            sql = "UPDATE student SET student_name = %s,cnic= %s,email=%s,student_no=%s,department=%s,section=%s WHERE student_id = %s"
            cursor.execute(sql,(s_name,s_cnic,s_email,s_no,s_department,s_section,student_id))
            mydb.commit()
            return redirect('/dashboard')      
    sql="select * from student where student_id = %s"  
    cursor.execute(sql,(student_id,))  
    s=cursor.fetchone()  
    return render_template('editstudent.html', params=params, s=s)
@app.route("/deletestudent/<string:student_id>", methods = ['GET', 'POST'])#saad
def deletestudent(student_id):
    if('user' in session and session['user']== params['admin_user']): 
        sql = "DELETE FROM student WHERE student_id = %s"
        cursor.execute(sql,(student_id,))
        mydb.commit()
        print(cursor.rowcount, "record(s) deleted")
    return redirect('/dashboard')
@app.route("/editinstructor/<string:teacher_id>", methods =['GET','POST']) #saad
def editinstructor(teacher_id):
    if('user' in session and session['user']== params['admin_user']):
        if request.method == 'POST':
            t_id= request.form.get('teacher_id')
            t_name= request.form.get('teacher_name')
            t_cnic= request.form.get('teacher_cnic')
            t_email= request.form.get('teacher_email')
            t_department= request.form.get('teacher_department')
            sql = "UPDATE instructor SET teacher_name = %s ,teacher_cnic=%s,teacher_email=%s,teacher_department=%s WHERE teacher_id = %s"
            cursor.execute(sql,(t_name,t_cnic,t_email,t_department,teacher_id))
            mydb.commit()
            return redirect('/instructor')
    sql="select * from instructor where teacher_id = %s"  
    cursor.execute(sql,(teacher_id,))  
    t=cursor.fetchone()  
    return render_template('editinstructor.html', params=params, t=t)
@app.route("/deleteinstructor/<string:teacher_id>", methods = ['GET', 'POST'])#saad
def deleleinstructor(teacher_id):
    if('user' in session and session['user']== params['admin_user']):
        sql = "DELETE FROM instructor WHERE teacher_id = %s"
        cursor.execute(sql,(teacher_id,))
        mydb.commit()
        print(cursor.rowcount, "record(s) deleted")   
    return redirect('/instructor')
@app.route("/attendance/<wek>")
def attendance(wek):

    now= datetime.now()
    dtStr= now.strftime('%I_%M_%S')
    if int(wek) > 0:
        weeks= "week " + str(wek)
        
        sql = "select * from attendance where(cour_id = %s && week= %s)"
        att= cursor.execute(sql,(courses,wek,)) 
        att=cursor.fetchall()
        print("in if")
        sql = "select course_name from courses where course_code= %s"
        cou= cursor.execute(sql,(courses,)) 
        cou=cursor.fetchall()
        for c in cou:
            c_name=c[0]
        
  
        return render_template('attendance.html',att=att,courses=courses,c_name=c_name,dtStr=dtStr,weeks=weeks)
    else:
        print(wek)
        sql = "select * from attendance"
        att= cursor.execute(sql) 
        att=cursor.fetchall()
        print("in else")
    
        return render_template('attendance.html',att=att,dtStr=dtStr,wek=wek)
@app.route("/panel")
def panel():
    pan="OFF"
    return redirect('/home')
@app.route("/generator")
def generator():
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
        st=str(75.0)
    return render_template('generator.html',attendance=attendance,st=st)
    

    
    
@app.route("/logout")
def logout():
    session.pop('user')
    return redirect('/')
app.run()

