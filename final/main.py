from matplotlib import image
import numpy as np
import base64
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
#from tensorflow.keras.preprocessing import image
import os
import pandas as pd 
import random
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import itertools
import h5py
import io
import pickle
from keras.models import load_model
from keras.models import Model
# Deep learning libraries
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras.models import model_from_json
#from tensorflow.keras.preprocessing import image
from streamlit_option_menu import option_menu
##code startes
with st.sidebar:
    selected = option_menu(None,
                          ['Home',
                          'Pneumonia',
                          'Malaria',
                           'Heart Disease',
                           'Diabetes(Women)',
                           ],
                          icons=['house-fill','bi bi-meta','thermometer-half','bi bi-heart','bi bi-capsule'],
                          default_index=0)

if selected=="Home":
     
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            st.markdown(
            f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
        }}
    </style>
    """,
        unsafe_allow_html=True
    )
    add_bg_from_local('images/home.png')
   
if selected=="Malaria":
    img1=Image.open("images/image-removebg-preview (32).png")
    st.image(img1)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    link_malaria ="https://en.wikipedia.org/wiki/Malaria#Society_and_culture"
    text_malaria="Click to learn more"
    
    label_malaria=f"Prediction of Malaria : [{text_malaria}]({link_malaria})"

    def load_cnn1():
        model_ = load_model('malaria.h5')
        return model_

    def preprocessed_image(file):
        image = file.resize((44,44), Image.ANTIALIAS)
        image = np.array(image)
        image = np.expand_dims(image, axis=0) 
        return image

    def main():
        st.title("Prediction of Malaria")
        link_malaria ="https://en.wikipedia.org/wiki/Malaria#Society_and_culture"
        st.write(f'[Click to learn more]({link_malaria})')
        
        model_1 = load_cnn1()
        images = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
        if images is not None:
            images = Image.open(images)
            st.text("Image Uploaded!")
            st.image(images,width=300)
            used_images = preprocessed_image(images)
            predictions = np.argmax(model_1.predict(used_images), axis=-1)
            if predictions == 1:
                st.error("The data affected with malaria")
            elif predictions == 0:
                st.success("The data is not affected with malaria")
                
    if __name__ == "__main__":
        main()

if selected=="Pneumonia":
    loaded_model=tf.keras.models.load_model('pnemonia.h5')
    img2=Image.open('images/image-removebg-preview (31).png')
    st.image(img2)
    st.title("Prediction of Pneumonia")
    link_pneu='https://en.wikipedia.org/wiki/Pneumonia'
    st.write(f'[Click to learn more]({link_pneu})')
    file=st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
    def predict(image_path):
        image1 = image.load_img(image_path, target_size=(150, 150))
        image1 = image.img_to_array(image1)
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        #st.write(image1.shape)
        img_array= image1/255
        prediction = loaded_model.predict(img_array)
        if prediction[0][0]>.6:
            st.error("The data affected with Pneumonia")
        else :
            st.success("The data is not affected with Pneumonia")
        
    if file is not None:
        img=Image.open(file).convert('RGB')
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        predict(file)


if selected=="Heart Disease":
    loaded_model=tf.keras.models.load_model('pnemonia.h5')
    img2=Image.open('images/ss.png')
    st.image(img2)
    pickle_in = open('heart.pkl','rb')
    heart = pickle.load(pickle_in)
    def prediction(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
        prediction = heart.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        print(prediction)
        return prediction
        # this is the main function in which we define our webpage
    def main():
            # giving the webpage a title
        st.title("Prediction of Heart Disease")
        link_heart='https://en.wikipedia.org/wiki/Cardiovascular_disease'
        st.write(f'[Click to learn more]({link_heart})')
            # the following lines create text boxes in which the user can enter
            # the data required to make the prediction
        age = st.slider("select your age",15,100)
        sex = st.selectbox("Sex",["Male","Female"])
        if sex=="Male":
            sex=1
        if sex=="Female" :
            sex=0
        link_url_1='https://en.wikipedia.org/wiki/Constrictive_pericarditis'
        link_text_1='Click to learn more'
        label_1=f"Constrictive pericarditis : [{link_text_1}]({link_url_1})"
        cp=st.radio(label_1,["Yes","No"])
        if cp=="Yes":
            cp=1
        if cp=="No":
            cp=0
        trestbps = st.slider("Heart Beat",45,180)
        link_url='https://en.wikipedia.org/wiki/Cholesterol'
        link_text='Click to learn more'
        label1=f"Cholestrol : [{link_text}]({link_url})"
        chol = st.slider(label1,100,300)
        link_url_2="https://my.clevelandclinic.org/health/diagnostics/21952-fasting-blood-sugar"
        label_2=f"Cholestrol : [{link_text}]({link_url_2})"
        fbs = st.slider(label_2,80,300)
        if fbs>=120:
            fbs =1
        if fbs<120:
            fbs=0
        link_url_3 ="https://www.ncbi.nlm.nih.gov/books/NBK367910/"
        label_3=f"Rest ecg: [{link_text}]({link_url_3})"
        restecg = st.slider(label_3,0,2)
        thalach = st.slider("Maximun Heart Rate",100,200)
        link_url_4 = "https://en.wikipedia.org/wiki/Angina"
        label_4=f"Exercise induced angina: [{link_text}]({link_url_4})"
        exang = st.radio(label_4,["Yes","No"])
        if exang =="Yes":
            exang=1
        if exang=="No":
            exang=0
        oldpeak = st.slider("Oldpeak",0.0,7.0)
        slope = st.slider("Slope",0,3)
        link_ca ="https://www.mayoclinic.org/diseases-conditions/coronary-artery-disease/symptoms-causes/syc-20350613"
        label_ca=f"Coronary Artery Disease: [{link_text}]({link_ca})"
        ca = st.slider(label_ca,0,3)
        link_tal = "https://www.healthline.com/health/thallium-stress-test"
        label_tal=f"Thalium stress result: [{link_text}]({link_tal})"
        thal = st.slider(label_tal,0,3)
            #target= st.text_input("target", "0 or 1")
        result =""
        if st.button("Predict"):
            result = prediction(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
            print(result)
            if result == 1:
                st.error("The data affected with Heart disease")
            else:
                st.success("The data is not affected with Heart disease")
    if __name__=='__main__':
        main()

if selected == "Diabetes(Women)":
    # loading in the model to predict on the data
    pickle_in = open('diabetes.pkl', 'rb')
    dia = pickle.load(pickle_in)

    def prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
        prediction =dia.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        print(prediction)
        return prediction
    def main():
        st.title("Prediction of Diabetes")
        link_dmenu="https://en.wikipedia.org/wiki/Diabetes"
        st.write(f'[Click to learn more]({link_dmenu})')
        Pregnancies = st.slider("No.of times Pregnancies ",1,10)
        Glucose = st.slider("Glucose level",125,320)
        BloodPressure=st.slider("Blood pressure",120,180)
        SkinThickness =st.slider("Skin_thickness",0,50)
        Insulin =st.slider("Insulin",0,850)
        BMI = st.slider("BMI",0,50)
        DiabetesPedigreeFunction = st.slider("DiabetesPedigreeFunction",0.0,1.0)
        Age=st.slider("Age",15,65)
        result =""
        if st.button("Predict"):
            result = prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
            print(result)
            if result == 1:
                st.error("Your are affected by Diabetics")
            else:
                st.success("your not affected by Diabetics")
    if __name__=='__main__':
	    main()
 
                
    
	
