from flask import Flask, render_template, request,Response,redirect
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_facenet import FaceNet
import numpy as np
import pickle
import cv2
import PIL
from PIL import Image
# from flask_sqlalchemy import SQLAlchemy
HaarCascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
facenet = FaceNet()
folder='Pictures/'
database_file = 'Database.pickle'
global database 
database = {}

global capture , VName , remove ,check
capture , remove ,check= 0,0,0
app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# db = SQLAlchemy(app)
# class Member(db.Model):
#     id = db.Column(db.Integer(),primary_key=True)
#     name = db.Column(db.String(),nullable=False)
#     no = db.Column(db.String(10),nullable=False)
#     address = db.Column(db.String(),nullable=False)
# with app.app_context():
#     db.create_all()
#     db.session.commit()
camera = cv2.VideoCapture(0)

def add():
    global capture
    capture=0
    success,Img=camera.read()
    Img = cv2.flip(Img,1)
    face = HaarCascade.detectMultiScale(Img,1.1,4)
    if len(face)>0:
        x1, y1, width, height = face[0]         
    else:
        x1, y1, width, height = 1, 1, 5, 5
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    Img1 = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    Img1 = PIL.Image.fromarray(Img1)                 
    Img_array = np.asarray(Img1)
    face = Img_array[y1:y2, x1:x2]                        
    face = PIL.Image.fromarray(face)                       
    face = face.resize((160,160))
    face = np.asarray(face)
    face = np.expand_dims(face, axis=0)
    embedd = facenet.embeddings(face)
    embedd = embedd / np.linalg.norm(embedd, ord=2)
    min_dist=0
    identity=' '
    for key, value in database.items() :
        dist = np.dot(value,embedd.T)
        if dist > min_dist:
            min_dist = dist
            identity = key
    if min_dist < 0.7:
        p = os.path.sep.join(['Pictures', "{}.jpg".format(str(VName).replace(":",''))])
        cv2.imwrite(p, Img)
        database[VName] = embedd
        with open(database_file, 'wb') as f:
            pickle.dump(database, f)


def see(Img,x1,y1,x2,y2):
    Img1 = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    Img1 = PIL.Image.fromarray(Img)                 
    Img_array = np.asarray(Img1)
    face = Img_array[y1:y2, x1:x2]                        
    face = PIL.Image.fromarray(face)                       
    face = face.resize((160,160))
    face = np.asarray(face)
    face = np.expand_dims(face, axis=0)
    embedd = facenet.embeddings(face)
    embedd = embedd / np.linalg.norm(embedd, ord=2)
    min_dist=0
    identity=' '
    for key, value in database.items() :
        dist = np.dot(value,embedd.T)
        if dist > min_dist:
            min_dist = dist
            identity = key
    Img = cv2.flip(Img,1)
    if min_dist < 0.7:
        cv2.putText(Img,'unknown', (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(Img,identity, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(Img,"            " +str(min_dist), (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return Img

def delete(Img,x1,y1,x2,y2):#function needs to be done
    global remove
    remove = 0
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    Img = PIL.Image.fromarray(Img)                 
    Img_array = np.asarray(Img)
    face = Img_array[y1:y2, x1:x2]                        
    face = PIL.Image.fromarray(face)                       
    face = face.resize((160,160))
    face = np.asarray(face)
    face = np.expand_dims(face, axis=0)
    embedd = facenet.embeddings(face)
    embedd = embedd / np.linalg.norm(embedd, ord=2)
    min_dist=0
    identity=' '
    for key, value in database.items() :
        dist = np.dot(value,embedd.T)
        if dist > min_dist:
            min_dist = dist
            identity = key
    if min_dist < 0.7:
         print("Identity cannot be found")
    else:
        del database[identity]
        p = os.path.sep.join(['Pictures', "{}.jpg".format(str(identity).replace(":",''))])
        os.remove(p)
        with open(database_file, 'wb') as f:
            print("File Dumped")
            pickle.dump(database, f)
        print (identity + ' deleted ')


def face_to_embeddings(Img):
    face = HaarCascade.detectMultiScale(Img,1.1,4)
    
    if len(face)>0:
        x1, y1, width, height = face[0]         
    else:
        x1, y1, width, height = 1, 1, 5, 5
        
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    Img = PIL.Image.fromarray(Img)                 
    Img_array = np.asarray(Img)
    
    face = Img_array[y1:y2, x1:x2]                        
    
    face = PIL.Image.fromarray(face)                       
    face = face.resize((160,160))
    face = np.asarray(face)
    
    face = np.expand_dims(face, axis=0)
    embedd = facenet.embeddings(face)
    embedd = embedd / np.linalg.norm(embedd, ord=2)
    return embedd, x1, y1, x2, y2

def generate_frames():
    global VName
    while True:
        global capture , check , remove
        ## read the camera frame
        success,Img=camera.read()
        face = HaarCascade.detectMultiScale(Img,1.1,4)
        if len(face)>0:
            x1, y1, width, height = face[0]         
        else:
            x1, y1, width, height = 1, 1, 5, 5
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        Img1 = Img
        cv2.rectangle(Img,(x1,y1),(x2,y2), (0,255,0), 2)
        Img = cv2.flip(Img,1)
        if(capture):
            add()
            # break
            #return
        if (check):
            Img = see(Img1,x1,y1,x2,y2)
        if (remove):
           s = delete(Img1,x1,y1,x2,y2)
        ret,buffer=cv2.imencode('.jpg',Img)
        Img=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + Img + b'\r\n')
        

@app.route('/home',methods=['POST','GET'])
def home():
    global check
    if request.method == 'POST':
        if request.form.get('verify') == 'verify':
            check = 1
    elif request.method == 'GET':
         return render_template('index.html')
    return render_template('index.html')

@app.route("/")
def Hello_world():
    global database
    if os.path.exists(database_file):
        with open(database_file, 'rb') as f:
            database = pickle.load(f)
    for filename in os.listdir(folder):
        if filename.split('.')[0] not in database.keys():
            path = folder + filename
            gbr1 = cv2.imread(path)
            signature, _, _, _, _ = face_to_embeddings(gbr1)
            database[os.path.splitext(filename)[0]] = signature
    with open(database_file, 'wb') as f:
        pickle.dump(database, f)
    return redirect('/home')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view')
def View():
    global check
    check = 0
    # items=Member.query.all()
    return render_template("view.html",database=enumerate(database.keys()))

@app.route('/register',methods=['POST','GET'])#show if the visitor is added or not
def Register():
    global check,capture,VName
    check = 0
    if request.method == 'POST':
        if request.form.get('register') == 'register':
            capture=1
            VName = request.form['VName']
            # mem=Member(name=request.form['VName'],no=request.form['number'],address=request.form['address'])
            # db.session.add(mem)
            # db.session.commit()
    elif request.method == 'GET':
         return render_template('register.html')
    return render_template('register.html')

@app.route('/delete',methods=['POST','GET'])#show if visitor is deleted or not
def Delete():
    global check,remove
    check = 0
    if request.method == 'POST':
        if request.form.get('delete') == 'delete':
            remove = 1
    elif request.method == 'GET':
         return render_template('delete.html')
    return render_template('delete.html')

if __name__ == '__main__':
    app.run(debug = True)