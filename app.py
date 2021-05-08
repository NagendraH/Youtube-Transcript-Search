from flask import Flask
from flask import Flask, redirect, url_for
from flask_ngrok import run_with_ngrok
from flask import render_template
from flask import request
import h5py
from flaskext.mysql import MySQL
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import pickle
from keras import backend as K
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adadelta

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
    
malstm=load_model('quora.h5')
app = Flask(__name__)

mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = "u9fwcv0v3gajj3ky"
app.config['MYSQL_DATABASE_PASSWORD'] = "qzgorFn5oXvosBhJs7t9"
app.config['MYSQL_DATABASE_DB'] = "bff29zhocbsecefuqe3p"
app.config['MYSQL_DATABASE_HOST'] = "bff29zhocbsecefuqe3p-mysql.services.clever-cloud.com"
mysql.init_app(app)
yid1=''
yid=''
temp1=''
#run_with_ngrok(app)
gradient_clipping_norm = 1.25
batch_size =1
n_epoch = 25
'''@app.route('/<string:insert>')
def add(insert):
    sql = "select *from user;"
    conn = mysql.connect()
    cursor =conn.cursor()
    cursor.execute(sql)
    data=cursor.fetchall()
    print(data)
    return render_template('display.html',data=data)'''
    
@app.route("/",methods=['post', 'get'])
def upload_predict():
    l={}
    global yid1
    global yid
    global temp1
    transcripts1=[]
    if request.method == 'POST':
        if request.form.get("submit_b"):
            option = " ".join(map(str,(request.form['options'])))
            print(option)
            print(yid)
            sql = "INSERT INTO user (question1, question2) VALUES (%s, %s);"
            val = (yid,option)
            #print(s1[i])
            conn = mysql.connect()
            cursor =conn.cursor()
            cursor.execute(sql, val)
            conn.commit()
            

        if request.form.get("submit_a"):
            data=YouTubeTranscriptApi.get_transcript(str(yid1[24:]),languages=['en', 'en-IN'])
            time1=[]
            transcripts1=[]
            for values in data:
                for key,value in values.items():
                    if(key=="text"):
                        transcripts1.append(value)
                    if(key=="start"):
                        time1.append(value)
        if(request.form.get('gsearch1')!=None):
            #print(yid1)
            yid1=str(request.form.get('gsearch1'))
            temp1='https://www.youtube.com/embed/'+yid1[24:]
        if(request.form.get('gsearch2')!=None):
            yid = request.form.get('gsearch2')
            #print(yid)
            data=YouTubeTranscriptApi.get_transcript(str(yid1[24:]),languages=['en', 'en-IN'])
            time=[]
            transcripts=[]
            for values in data:
                for key,value in values.items():
                    if(key=="text"):
                        transcripts.append(value)
                    if(key=="start"):
                        time.append(value)
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            q1 = tokenizer.texts_to_sequences(np.array(transcripts))
            q1 = pad_sequences(q1, maxlen = 30, padding='post')
            q3 = tokenizer.texts_to_sequences(np.array([str(yid)]))
            q3 = pad_sequences(q3, maxlen = 30, padding='post')
            p=[]
            for i in range(len(q1)):
                p.append(malstm.predict([q3[0],q1[i]]))
            s=[]
            print(q3[0])
            for i in range(len(p)):
                if(np.mean(np.array(p[i]))>0.1):
                    #print(transcripts[i],np.mean(np.array(p[i])))
                    s.append(np.mean(np.array(p[i])))
            sort_index = np.argsort(s) 
            sort_index=list(sort_index)
            sort_index.reverse()
            l={}
            for i in sort_index[:4]:
                l[transcripts[i]]=(temp1+'?autoplay=1&start='+str(round(time[i])))
                
    print(temp1)
    return render_template('index.html', result = l,id=temp1,transcripts1=transcripts1)
    

if __name__=="__main__":
  app.run()