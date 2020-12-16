from flask import Flask, render_template, request, session, redirect
from flask_sqlalchemy import SQLAlchemy
import json
import os
from datetime import datetime
import face_recognition
import imutils
import pickle
import time
import cv2
import os
from datetime import datetime
import pyttsx3 

with open('config.json', 'r') as c:
    params = json.load(c)["params"]

local_server = True
app = Flask(__name__)
app.secret_key='saad'
if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']

db = SQLAlchemy(app)

class Attendance(db.Model):
    course_id = db.Column(db.Integer,nullable=False )
    student_id = db.Column(db.String(50), primary_key=True)
    Section = db.Column(db.String(50), nullable=False)
    week= db.Column(db.String(10), nullable=False)
    datetime = db.Column(db.String(12), nullable=True)


class Instructor(db.Model):
    TeacherID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(50), nullable=False)

class Courses(db.Model):
    ID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(50), nullable=False)
    
class Student(db.Model):
    student_id= db.Column(db.Integer, primary_key=True)
    student_name= db.Column(db.String(50), nullable=False)
    Section= db.Column(db.String(50), nullable=False)
   
          

@app.route("/") #saad
def main():
    return render_template('main.html', params=params)

@app.route("/home")#saad
def home():
    
    return render_template('home.html', params=params)


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
    if('user' in session and session['user']== params['admin_user']):
        
        posts = User.query.all()
        return render_template('doctors.html', params=params, posts=posts)
    
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
            global coursename
        
            id= int(courses)
            post = Courses.query.filter_by(ID=id).first()
            if id == post.ID:
                    coursename=post.Name
                    
            else:
                return redirect('/home')
            stu= Student.query.filter_by(student_name=n).first()
            if n == stu.student_name:
                    name=stu.student_name
                    print(name)
                    comp=week
                    at=Attendance.query.filter_by(student_id=stu.student_name).first()
                    print(at)

                    if at:
                        
                        if (at.week == comp):
                            
                            print("Already")
                        else:
                            
                                
                            now= datetime.now()
                            dtString= now.strftime('%I_%M_%S_%p')
                            entry = Attendance(course_id= post.ID,student_id= name,Section= stu.Section,week =comp,datetime =dtString)
                            db.session.add(entry)
                            db.session.commit()
                       
                    else:
                        
                        now= datetime.now()
                        dtString= now.strftime('%I_%M_%S_%p')
                        entry = Attendance(course_id= post.ID,student_id= name,Section= stu.Section,week =comp,datetime =dtString)
                        db.session.add(entry)
                        db.session.commit()
                        print("Attendance Marked")
                        
                    


                         
            else:
                tec=Instructor.query.filter_by(Name=n).last()
                if n == tec.Name:
                    name=tec.name
                    relay(name)
                else:
                    print("not found")
                    
        
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
    relay
        
                    
            
                    
                    
                    
        
        
        
        
        
    return redirect('/home')

@app.route("/index", methods = ['GET', 'POST']) #saad
def index():
    uname=""
    if('user' in session and session['user']== uname):
        
        return redirect('/home')
    
    if request.method=="POST":
        username= request.form.get('email')
        password= request.form.get('password')
        posts = Instructor.query.all()
        for post in posts: 
            if(username == post.Name  and password ==post.password ):
                
                #set the session variable
                session['user'] = post.Name
                uname=post.Name
                return redirect('/home')
    
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

@app.route("/editdoctor/<string:user_id>", methods =['GET','POST'])
def editdoctor(user_id):
    if('user' in session and session['user']== params['admin_user']):
        if request.method == 'POST':
            boxpassword= request.form.get('password')
            boxname= request.form.get('name')
            boxemail= request.form.get('email')
            boxcnic= request.form.get('cnic_no')
            boxcontact= request.form.get('contact_no')
            boxgender= request.form.get('gender')
            boxaddress= request.form.get('address')
            boxage= request.form.get('age')
          
           
            
            post=User.query.filter_by(user_id=user_id).first()
            post.password =  boxpassword
            post.name = boxname
            post.email = boxemail
            post.cnic_no = boxcnic
            post.contact_no = boxcontact
            post.gender = boxgender
            post.address= boxaddress
            post.age = boxage
    
            db.session.commit()
            return redirect('/editdoctor/'+user_id)
        
    post=User.query.filter_by(user_id=user_id).first()       
    return render_template('editdoctor.html', params=params, post=post)


@app.route("/deletepatient/<string:patient_id>", methods = ['GET', 'POST'])
def deletepatient(patient_id):
    if('user' in session and session['user']== params['admin_user']):
        
        post=Patient.query.filter_by(patient_id=patient_id).first()
        db.session.delete(post)
        db.session.commit()
    
    return redirect('/patient')

@app.route("/deletedoctor/<string:user_id>", methods = ['GET', 'POST'])
def deletedoctor(user_id):
    if('user' in session and session['user']== params['admin_user']):
        
        post=User.query.filter_by(user_id=user_id).first()
        db.session.delete(post)
        db.session.commit()
    
    return redirect('/doctors')

@app.route("/signup", methods = ['GET', 'POST'])
def signup():
    if(request.method=='POST'):
        password = request.form.get('password')
        name = request.form.get('name')
        email = request.form.get('email')
        cnic_no = request.form.get('cnic_no')
        contact_no = request.form.get('contact_no')
        gender = request.form.get('gender')
        address = request.form.get('address')
        age = request.form.get('age')
        
        
        entry = User(password= password,name= name,email= email,cnic_no= cnic_no,contact_no= contact_no,gender= gender,address= address,age= age)
        db.session.add(entry)
        db.session.commit()
        
        
        
    return redirect('/index')

@app.route("/siteterms")
def siteterms():
    return render_template('site terms.html', params=params)

@app.route("/developers")
def developers():
    return render_template('developers.html', params=params)

@app.route("/aibpds")
def aibpds():
    return render_template('aibpds.html', params=params)

@app.route("/services")
def services():
    return render_template('services.html', params=params)

@app.route("/faq")
def faq():
    return render_template('faq.html', params=params)

@app.route("/galary")
def galary():
    return render_template('galary.html', params=params)

@app.route("/contact")
def contact():
    return render_template('contact.html', params=params)

@app.route("/uploadingimage")
def uploadingimage():
    return render_template('uploadingimage.html', params=params)

@app.route("/a1")
def a1():
    return render_template('a1.html', params=params)

@app.route("/a2")
def a2():
    return render_template('a2.html', params=params)

@app.route("/a3")
def a3():
    return render_template('a3.html', params=params)

@app.route("/a4")
def a4():
    return render_template('a4.html', params=params)

@app.route("/a5")
def a5():
    return render_template('a5.html', params=params)

@app.route("/a6")
def a6():
    return render_template('a6.html', params=params)

@app.route("/uploader", methods = ['GET', 'POST'])
def uploader():
    
    if request.method=="POST":
        
        f=request.files['file1']
        print(f)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        print(f.filename)
        global req
        req= f.filename
        print("##############################################")
        print(req)
        return redirect('/aimodel')
    
@app.route("/uload")
def uload():
    src = "C:\\Users\\Shabbir&Sons\\Desktop\\keras-frcnn\\results_imgs\\"
    dst = "C:\\Users\\Shabbir&Sons\\Desktop\\keras-frcnn\\static\\resultimg\\"
    f=res
    shutil.copy(path.join(src, f), dst)
    
        
  
    
    return redirect('/output')
 

@app.route("/aimodel")
def aimodel():
    
    
    
    sys.setrecursionlimit(40000)
    
    
    
    config_output_filename = "config.pickle"
    
    
    
    with open(config_output_filename, 'rb') as f_in:
    	C = pickle.load(f_in)
    	K.clear_session()
        
    
    if C.network == 'resnet50':
        import keras_frcnn.resnet as nn
    	
    elif C.network == 'vgg':
        import keras_frcnn.vgg as nn
    	
    
    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
    
    img_path = "test_images"
    print("************************************************")
    print(img_path)
    print("************************************************")
    
    def format_img_size(img, C):
    	""" formats the image size based on config """
    	img_min_side = float(C.im_size)
    	(height,width,_) = img.shape
    		
    	if width <= height:
    		ratio = img_min_side/width
    		new_height = int(ratio * height)
    		new_width = int(img_min_side)
    	else:
    		ratio = img_min_side/height
    		new_width = int(ratio * width)
    		new_height = int(img_min_side)
    	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    	return img, ratio	
    
    def format_img_channels(img, C):
    	""" formats the image channels based on config """
    	img = img[:, :, (2, 1, 0)]
    	img = img.astype(np.float32)
    	img[:, :, 0] -= C.img_channel_mean[0]
    	img[:, :, 1] -= C.img_channel_mean[1]
    	img[:, :, 2] -= C.img_channel_mean[2]
    	img /= C.img_scaling_factor
    	img = np.transpose(img, (2, 0, 1))
    	img = np.expand_dims(img, axis=0)
    	return img
    
    def format_img(img, C):
    	""" formats an image for model prediction based on config """
    	img, ratio = format_img_size(img, C)
    	img = format_img_channels(img, C)
    	return img, ratio
    
    # Method to transform the coordinates of the bounding box to its original size
    def get_real_coordinates(ratio, x1, y1, x2, y2):
    
    	real_x1 = int(round(x1 // ratio))
    	real_y1 = int(round(y1 // ratio))
    	real_x2 = int(round(x2 // ratio))
    	real_y2 = int(round(y2 // ratio))
    
    	return (real_x1, real_y1, real_x2 ,real_y2)
    
    class_mapping = C.class_mapping
    
    if 'bg' not in class_mapping:
    	class_mapping['bg'] = len(class_mapping)
    
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0,255,3) for v in class_mapping}
    
    C.num_rois = 32
    print(C.num_rois)
    print("*******************************************")
    if C.network == 'resnet50':
    	num_features = 1024
    elif C.network == 'vgg':
    	num_features = 512
    
    if K.image_dim_ordering() == 'th':
    	input_shape_img = (3, None, None)
    	input_shape_features = (num_features, None, None)
    else:
    	input_shape_img = (None, None, 3)
    	input_shape_features = (None, None, num_features)
    
    
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)
    
    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)
    
    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
 
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)
    
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    
    model_classifier = Model([feature_map_input, roi_input], classifier)
    
    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)
    
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
    
    all_imgs = []
    
    classes = {}
    
    bbox_threshold = 0.8
    
    visualise = True
    img_name_list = []
    lists = []
    # req = "abc.png"
    
    
    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
    		continue
    	img_name=req 
        
        
    	img_name_list.append(img_name)
    	#print(img_name_list)
    	st = time.time()
    	filepath = os.path.join(img_path,img_name)
    
    	img = cv2.imread(filepath)
    
    	X, ratio = format_img(img, C)
    
    	if K.image_dim_ordering() == 'tf':
    		X = np.transpose(X, (0, 2, 3, 1))
    
    	# get the feature maps and output from the RPN
    	[Y1, Y2, F] = model_rpn.predict(X)
    
    	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
    
    	# convert from (x1,y1,x2,y2) to (x,y,w,h)
    	R[:, 2] -= R[:, 0]
    	R[:, 3] -= R[:, 1]
    
    	# apply the spatial pyramid pooling to the proposed regions
    	bboxes = {}
    	probs = {}
    
    	for jk in range(R.shape[0]//C.num_rois + 1):
    		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
    		if ROIs.shape[1] == 0:
    			break
    
    		if jk == R.shape[0]//C.num_rois:
    			#pad R
    			curr_shape = ROIs.shape
    			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
    			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
    			ROIs_padded[:, :curr_shape[1], :] = ROIs
    			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
    			ROIs = ROIs_padded
    
    		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
    
    		for ii in range(P_cls.shape[1]):
    
    			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
    				continue
    
    			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
    
    			if cls_name not in bboxes:
    				bboxes[cls_name] = []
    				probs[cls_name] = []
    
    			(x, y, w, h) = ROIs[0, ii, :]
    
    			cls_num = np.argmax(P_cls[0, ii, :])
    			try:
    				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
    				tx /= C.classifier_regr_std[0]
    				ty /= C.classifier_regr_std[1]
    				tw /= C.classifier_regr_std[2]
    				th /= C.classifier_regr_std[3]
    				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
    			except:
    				pass
    			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
    			probs[cls_name].append(np.max(P_cls[0, ii, :]))
    
    	all_dets = []
    	
    	#i = 0
    
    	for key in bboxes:
    		bbox = np.array(bboxes[key])
    
    		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
    		for jk in range(new_boxes.shape[0]):
    			(x1, y1, x2, y2) = new_boxes[jk,:]
    
    			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
    			#bbx_df = pd.DataFrame((real_x1, real_y1, real_x2, real_y2))
    			
    
    			#print("X1 ",real_x1)
    			#print("Y1 ",real_y1)
    			#print("X2 ",real_x2)
    			#print("Y2 ",real_y2)
    
    			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
    
    			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
    			all_dets.append((key,100*new_probs[jk]))
    
    			lists.append([real_x1,real_y1,img_name,idx,key])
    
    			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.4,1)
    			textOrg = (real_x1, real_y1-0)
                
                #cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), color = None)
    			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 150, textOrg[1]-retval[1] - 45), (255, 255, 255), -1)
    			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 2)
    
    			#cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), color = None)
    			#cv2.rectangle(img, (textOrg[0] - 3,textOrg[1]+baseLine - 3), (textOrg[0]+retval[0], textOrg[1]-retval[1]), (255,255,255), -1)
    			#cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255))
    
    	print('Elapsed time = {}'.format(time.time() - st))
    	print(all_dets)
    	#cv2.imshow('img', img)
    	#cv2.waitKey(0)
    	cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
        
    	bbx_df = pd.DataFrame(lists)
        
    	bbx_df.to_csv('bbx_df.csv', header=None, index=None, sep=',')
    	break
    
    
    lIndex = req.rfind(".")
    global res 
    global domain
    domain= req[lIndex::]
    print(domain)
    res= "0"+req[lIndex::]
    print(res)
    K.clear_session()
   




    
   
    return redirect('/uload')  
 
@app.route("/output")
def result():
    
    values=[""]
    data= pd.read_csv("C:\\Users\\Shabbir&Sons\\Desktop\\keras-frcnn\\bbx_df.csv")
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print(data)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    df = pd.DataFrame(data)
    la=df.columns.tolist()  
    print("@@@@@@@@@@@@@@@@@@@@@@@ data column @@@@@@@@@@@@@@@@@")
    values.append(la[4])
    print(la)
    li=df.values.tolist()
    print(df)
    print(li)
    print("-----------------------------------")
            
    flag=0
    for val in li:
        
        print(li[flag][4])
        values.append(li[flag][4])
        flag=flag+1
    print("????????????????????")
    print(values)
    values.pop(0)
    print(values)
    global fresult
    global action
    for y in values:
        if y=="N":
            fresult="Normal or others"
            action="no action to perform"
            break
    for y in values:
        if y=="NT":
            fresult="Non-Tension Pneumothorax"
            action="immediate needle decompression by inserting a large-bore of 14- or 16-gauge needle into the 2nd intercostal space in the midclavicular line. Air will usually gush out."
            break 
    for x in values:
        if x=='T':
            fresult="Tension Pneumothorax"
            action="immediate needle decompression by inserting a large-bore of 14- or 16-gauge needle into the 2nd intercostal space in the midclavicular line. Air will usually gush out."
            break
    
           
    print(fresult)
    print(action)
    
    return render_template('result.html', params=params,res=res,fresult=fresult,action=action)
    
@app.route("/emailapi")
def emailapi():
    return render_template('emailapi.html', params=params)
    
@app.route("/patientform", methods = ['GET', 'POST'])
def patientform():
     global imagedb
     if(request.method=='POST'):
         imagedb =request.form.get('pname')
         print(imagedb)
         src = "C:\\Users\\Shabbir&Sons\\Desktop\\keras-frcnn\\static\\resultimg\\"
         dst = "C:\\Users\\Shabbir&Sons\\Desktop\\keras-frcnn\\static\\imagedb\\"
         f=res
         shutil.copy(path.join(src, f), dst)
         os.rename("C:\\Users\\Shabbir&Sons\\Desktop\\keras-frcnn\\static\\imagedb\\"+res,"C:\\Users\\Shabbir&Sons\\Desktop\\keras-frcnn\\static\\imagedb\\"+imagedb+domain)
         imagedb= imagedb+domain
         return render_template('patientform.html', params=params,imagedb=imagedb,fresult=fresult)
         
         

@app.route("/patientsignup", methods = ['GET', 'POST'])
def patientsignup():
    if(request.method=='POST'):
        
    
        name = request.form.get('name')
        contact_no = request.form.get('contact_no')
        gender = request.form.get('gender')
        address = request.form.get('address')
        email = request.form.get('email')
        age = request.form.get('age')
        cnic_no = request.form.get('cnic_no')
        xray = request.form.get('upload_xray')
        result_predicted = request.form.get('result_predicted')
        
        
        entry = Patient(name= name,contact_no= contact_no,gender= gender,address= address,email= email,age= age,cnic_no =cnic_no,xray = xray,result_predicted= result_predicted)
        db.session.add(entry)
        db.session.commit()
        
    return redirect('/output')

@app.route("/logout")
def logout():
    session.pop('user')
    return redirect('/')




















@app.route("/about")
def about():
    return render_template('about.html', params=params)


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



        
    

@app.route("/contacts", methods = ['GET', 'POST'])
def contacts():
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
        
        
    return render_template('contacts.html', params=params)


app.run()

