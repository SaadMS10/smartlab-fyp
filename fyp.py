from flask import Flask, render_template, request, session, redirect
from flask_sqlalchemy import SQLAlchemy
import json
import os
from datetime import datetime


with open('config.json', 'r') as c:
    params = json.load(c)["params"]

local_server = True
app = Flask(__name__)

if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']

db = SQLAlchemy(app)

class Attendance(db.Model):
    course_id = db.Column(db.Integer,nullable=False )
    student_id = db.Column(db.String(50), primary_key=True)
    Section = db.Column(db.String(50), nullable=False)
    datetime = db.Column(db.String(12), nullable=True)
    week= db.Column(db.String(10), nullable=False)


class Instructor(db.Model):
    TeacherID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(50), nullable=False)
    
   
    
class Posts(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    slug = db.Column(db.String(50), nullable=False)
    content = db.Column(db.String(120), nullable=False)
    img_file=db.Column(db.String(50), nullable=False)
    date = db.Column(db.String(12), nullable=True)
    

@app.route("/")
def main():
    return render_template('main.html', params=params)


@app.route("/loginadmin", methods = ['GET', 'POST'])
def loginadmin():
    
    if('user' in session and session['user']== params['admin_user']):
        
        posts = Patient.query.all()
        return render_template('patient.html', params=params,posts=posts)
    
    if request.method=="POST":
        username= request.form.get('email')
        password= request.form.get('password')
        if(username == params['admin_user'] and password == params['admin_pass']):
            #set the session variable
            session['user'] = username
            
            posts = Patient.query.all()
           
            return render_template('patient.html', params=params, posts=posts)
    
    return render_template('login.html', params=params)

@app.route("/patient")
def patient():
    if('user' in session and session['user']== params['admin_user']):
        
        posts = Patient.query.all()
        return render_template('patient.html', params=params,posts=posts)
    
    
   

@app.route("/doctors")
def doctors():
    return render_template('doctors.html', params=params)

@app.route("/index", methods = ['GET', 'POST'])
def index():
    
    if('user' in session and session['user']== params['admin_user']):
        
        posts = Posts.query.all()
        return render_template('login.html', params=params,posts=posts)
    
    if request.method=="POST":
        username= request.form.get('email')
        password= request.form.get('password')
        posts = Instructor.query.all()
        for post in posts:
            if(username == post['Name'] and password == post['password']):
            #set the session variable
            session['user'] = username
            
            
            return render_template('login.html', params=params, posts=posts)
    
    return render_template('index.html', params=params)


@app.route("/editpatient/<string:patient_id>", methods =['GET','POST'])
def editpatient(patient_id):
    if('user' in session and session['user']== params['admin_user']):
        if request.method == 'POST':
            boxname= request.form.get('name')
            boxcontact= request.form.get('contact_no')
            boxgender= request.form.get('gender')
            boxaddress= request.form.get('address')
            boxemail= request.form.get('email')
            boxage= request.form.get('age')
            boxcnic= request.form.get('cnic_no')
            boxxray= request.form.get('xray')
            boxresult= request.form.get('result_predicted')
           
            
            post=Patient.query.filter_by(patient_id=patient_id).first()
            post.name = boxname
            post.contact_no = boxcontact
            post.gender = boxgender
            post.address= boxaddress
            post.email = boxemail
            post.age = boxage
            post.cnic_no = boxcnic
            post.xray= boxxray
            post.result_predicted = boxresult
            db.session.commit()
            return redirect('/editpatient/'+patient_id)
        
    post=Patient.query.filter_by(patient_id=patient_id).first()       
    return render_template('editpatient.html', params=params, post=post)

@app.route("/deletepatient/<string:patient_id>", methods = ['GET', 'POST'])
def deletepatient(patient_id):
    if('user' in session and session['user']== params['admin_user']):
        
        post=Patient.query.filter_by(patient_id=patient_id).first()
        db.session.delete(post)
        db.session.commit()
    
    return redirect('/patient')

@app.route("/logout")
def logout():
    session.pop('user')
    return redirect('/loginadmin')














@app.route("/about")
def about():
    return render_template('about.html', params=params)

@app.route("/uploader", methods = ['GET', 'POST'])
def uploader():
    if('user' in session and session['user']== params['admin_user']):
    
        if request.method=="POST":
            f=request.files['file1']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            return "Uploaded Successfully"






@app.route("/dashboard", methods = ['GET', 'POST'])
def dashboard():
    
    if('user' in session and session['user']== params['admin_user']):
        
        posts = Posts.query.all()
        return render_template('dashboard.html', params=params,posts=posts)
    
    if request.method=="POST":
        username= request.form.get('uname')
        password= request.form.get('pass')
        if(username == params['admin_user'] and password == params['admin_pass']):
            #set the session variable
            session['user'] = username
            
            posts = Posts.query.all()
            return render_template('dashboard.html', params=params, posts=posts)
    
    return render_template('login.html', params=params)

@app.route("/post")
def post():
    return render_template('post.html', params=params)

@app.route("/post/<string:post_slug>", methods =['GET'])
def post_route(post_slug):
    post= Posts.query.filter_by(slug=post_slug).first()
    return render_template('post.html', params=params, post=post)



        
    

@app.route("/contact", methods = ['GET', 'POST'])
def contact():
    if(request.method=='POST'):
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        entry = Contacts(name=name, phone_num = phone, msg = message, date= datetime.now(),email = email )
        db.session.add(entry)
        db.session.commit()
        mail.send_message('New message from ' + name,
                          sender=email,
                          recipients = [params['gmail-user']],
                          body = message + "\n" + phone
                          )
        
        
    return render_template('contact.html', params=params)

if __name__ == "__main__":
    app.run()
    app.run(debug=True)

