# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:10:25 2020

@author: SAAD

"""
import face_recognition
import imutils
import pickle
import time
import cv2
import os
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import pyttsx3 

# initialisation 
engine = pyttsx3.init() 
import mysql.connector
from datetime import datetime
from mysql.connector import Error
from flask import (
    Flask,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for
)

mydb = mysql.connector.connect(
        host="localhost", user="root",password="",database="fyp" )
cursor = mydb.cursor()

app = Flask(__name__)

app.secret_key='saad'

@app.route("/login", methods = ['GET', 'POST'])# saad
def instructor_login():
    global inst
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
            
            if(username1 == x[1] and password1 == x[2]):
                session['user'] = username1
                return redirect('/home')

    
    return render_template('login.html')
@app.route("/model", methods = ['GET', 'POST'])# saad
def model():
    
    if(request.method=='POST'):
        global courses
 
        courses = request.form.get('courses')
        week = request.form.get('week')

        engine = pyttsx3.init() 
        def relay(a):
            print("PANEL "+ a)
        cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
        faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
        data = pickle.loads(open('face_encoding', "rb").read())
        def mark_attendance(n):
            
            try:
                
                now = datetime.now()
                nameList=[]
                sql = """select Name from courses where ID = %s"""
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
                print(name)
                print("n"+n)
                sql = """select student_name from student where student_name = %s"""
                user= cursor.execute(sql, (name,))
                res=cursor.fetchall()
                print(res)
                if(res == []):
                    
                    sql = """select Name from instructor where Name = %s"""
                    user= cursor.execute(sql, (name,))
                    res=cursor.fetchall()   
                    if res != []:
                        incharge=[]
                        
                        if name not  in incharge:
                            print("----------------------------")
                            print("Lab Incharge " +name+ " Arrived")
                            engine.say("Lab Incharge " +name+ " Arrived") 
                            relay("ON")
                            time.sleep(5)
                            incharge.append(name)
                        else:
                            relay("OFF")
                            time.sleep(5)
                            incharge.remove(name)
                    else:
                        ("NOTHING")
                else:        
                    sql = """select student_id from attendance where (student_id = %s && week = %s)"""
                    user= cursor.execute(sql, (name,week))
                    res1=cursor.fetchall()
                    print(res1)
                    if res1 ==[]:
                        sql = """INSERT INTO attendance (course_id,student_id,Section,week,datetime)  VALUES (%s,%s,%s,%s,%s)"""
                        val=(courses,name,'B',week,dtString)
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
# loop over frames from the video file stream
        while True:
    # grab the frame from the threaded video stream
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convert the input frame from BGR to RGB 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
            encodings = face_recognition.face_encodings(rgb)
            names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
            for encoding in encodings:
                
                matches = face_recognition.compare_faces(data["encodings"],encoding)
        #set name =inknown if no encoding matches
                name = "Unknown"
        # check to see if we have found a match
                if True in matches:
                    
            #Find positions at which we get True and store them
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
                    for i in matchedIdxs:
                        
                        
                #Check the names at respective indexes we stored in matchedIdxs
                        name = data["names"][i]
                #increase count for the name we got
                        counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
                    name = max(counts, key=counts.get)
 
 
        # update the list of names
                names.append(name)
        # loop over the recognized faces
                for ((x, y, w, h), name) in zip(faces, names):
                    
            # rescale the face coordinates
            # draw the predicted face name on the image
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
        
                    
            
                    
                    
                    
        
        
        
        
        
    return redirect('/home')

@app.route("/home", methods = ['GET', 'POST']) #saad
def home():

    
    
    return render_template('home.html',inst=inst)
    


if __name__ == "__main__":
    app.run()
    app.run(debug=True)