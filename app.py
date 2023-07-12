
import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))
import pandas as pd

df = pd.read_excel("hii.xlsx")

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]]).astype(np.float64)
    prediction = model.predict_proba(input_data)
    pred = int(round(prediction[0][0]))
    return pred


def main():
    st.title("Crop Prediction")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Enter values to predict the suitable crop </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    N = st.text_input("N (Nitrogen)", "")
    P = st.text_input("P (Phosphorus)", "")
    K = st.text_input("K (Potassium)", "")
    temperature = st.text_input("Temperature", "")
    humidity = st.text_input("Humidity", "")
    ph = st.text_input("pH", "")
    rainfall = st.text_input("Rainfall", "")
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
       </div>
    """

    if st.button("Predict"):
        output = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        filtered_df = df.loc[df['class'] == output]
        crop = filtered_df['crop']
        
       
        
        st.success('The suitable crop is {}'.format(crop.to_string(index = False).upper()))
        

       

if __name__=='__main__':
    main()