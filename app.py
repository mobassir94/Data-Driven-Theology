# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:39:33 2022

@author: MOBASSIR
"""
from flask import Flask, render_template, url_for, request
import pandas as pd
from inference_utils import Multilingual_Quran_Bible_Search_Engine

# Cleaning the texts
#import re

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    
    if request.method == 'POST':
        query = request.form['comment']
    n_pairs = int(request.form.get('show_top_results'))
    

    if request.form.get('predict'):
        mlt_quran_bible =Multilingual_Quran_Bible_Search_Engine(query,size=n_pairs,language = 'en',metric = 'dot')
    elif request.form.get('predict1'):
        mlt_quran_bible = Multilingual_Quran_Bible_Search_Engine(query,size=n_pairs,language = 'en',metric = 'l2')
     
        
        
    return render_template('result.html', prediction=mlt_quran_bible)

if __name__ == '__main__':
    app.run(debug=False, port=33507)


