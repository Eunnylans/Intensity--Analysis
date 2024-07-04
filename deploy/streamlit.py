import base64
import os
import pickle
import re

import joblib
import pandas as pd
import streamlit as st
from text_normalization import text_normalizer

print(os.getcwd())

# Determine the directory of the current script
#current_dir = os.path.dirname(__file__)

# Define the relative paths to your .pkl files
#xgb_model_path = os.path.join(current_dir, 'pkl files', 'best_xgb_model.pkl')
#bow_vectorizer_path = os.path.join(current_dir, 'pkl files', 'bow_vectorizer.pkl')

# Load the XGBoost model
xgb_opt = joblib.load(r'C:\Users\eunic\OneDrive\Desktop\Knowledgehut\18 Capstone Project Guidelines\Intensity-Analysis-EmotionClassification\Intensity--Analysis\deploy\best_xgb_model.pkl')


# Load the BoW vectorizer
bow_vectorizer = joblib.load(r'C:\Users\eunic\OneDrive\Desktop\Knowledgehut\18 Capstone Project Guidelines\Intensity-Analysis-EmotionClassification\Intensity--Analysis\deploy\bow_vectorizer.pkl')


# Streamlit app
def main():

    #st.set_page_config(page_title="Emotion Classification", page_icon=":smiley:", layout="wide")
        
    # Inject CSS styles
    #def get_img_as_base64("")
    
    page_bg_img = f"""
    
    <style> 
            [data-testid="stAppViewContainer"] > .main {{
            background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQmVaIcgBOm4Ewy6FT095vv833WXHvLnfSvOrpsB8lsMFPSFLqGL7wDxq8daqsoH7pg_G0&usqp=CAU");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: local;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
                }}
                
            [data-testid="stHeader"] {{
                background: #cdf7f7;
                color: #0cbfae;
            }}
        
    
            [data-testid="stToolbar"] {{
            background: rgba(0,0,0,0);
            }}
    
            [data-baseweb="base-input"] {{
            background-color:#0cbfae;
            color: white;
            border: 4px solid white;
            }}
            
            [data-baseweb="base-input"]:focus {{
            background-color:#0cbfae;
            color: white;
            border: none;
            }}
            
            [data-testid="baseButton-secondary"] {{
            background-color:#0cbfae;
            color: white;
            }}
    
            [data-testid="stSidebar"] > div:first-child {{
            background-color:#0cbfae;
            text-align: justify;
            }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
   
st.markdown(
        """
        <div style="background-color:#0cbfae;padding:12px;border-radius:12px">
        <h1 style="color:white;text-align:center;">Emotion Prediction App <br/> 😊 😠 😢</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.write("## Enter Text for Emotion Prediction")

    # Text input
input_text = st.text_area("Enter your text here:")

if st.button("Predict"):
        if input_text:
            # Preprocess the input text
            preprocessed_text = text_normalizer(input_text)
            # Transform the text using the loaded vectorizer
            vectorized_text = bow_vectorizer.transform([preprocessed_text])
            # Predict the emotion
            prediction = xgb_opt.predict(vectorized_text)

            label_to_emotion = {
                0: ("happiness", "😊"),
                1: ("angry", "😠"),
                2: ("sad", "😢")
            }
            emotion, emoji = label_to_emotion[prediction[0]]

            # Display the result
            st.write(f"### Predicted Emotion: {emotion} {emoji}")
        else:
            st.error("Please enter some text for prediction.")

    # Add a sidebar for additional information or options
st.sidebar.header("About")
st.sidebar.info(
        """
        This application leverages a pretrained XGBoost model to accurately predict emotions from textual input. 
        The model has been meticulously trained on a diverse dataset encompassing various emotional states. Advanced 
        text preprocessing techniques have been employed to enhance the model's prediction accuracy, ensuring reliable 
        and insightful emotion analysis.
        """
    )

st.sidebar.header("Instructions")
st.sidebar.info(
        """
        1. Open the App
        2. Enter the text you want to analyze in the text box.
        3. Click the 'Predict' button to see the predicted emotion.
        """
    )

if __name__ == '__main__':
    main()
