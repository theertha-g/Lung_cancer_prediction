import streamlit as st
import pickle
from PIL import Image
from pyexpat import features


def main():
    st.title(":blue[LUNG CANCER PREDICTION]")
    img=Image.open('lungs.jpg')
    st.image(img,width=500)
    age=st.text_input("AGE")
    smoke=st.text_input("Average number of ciggerattes smoke in a day ")
    areaq=st.text_input("Quality of living area")
    alcohol=st.text_input("Average alcoholic consumption in a day")
    features=[age,smoke,areaq,alcohol]
    scaler=pickle.load(open("scaler.sav",'rb'))
    model=pickle.load(open("knn_model.sav",'rb'))
    pred=st.button('PREDICT')
    if pred:
        result=model.predict(scaler.transform([features]))
        if result==0:
            st.write("the person don't have lung cancer")
        else:
            st.write("the person is suffering lung cancer")
main()