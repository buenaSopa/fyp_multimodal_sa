import streamlit as st 
from textblob import TextBlob 
from transformers import pipeline
import numpy as np
from streamlit_shap import st_shap
from openai import OpenAI
import os
from dotenv import load_dotenv
import shap

load_dotenv()
shap.initjs()

@st.cache
def predict_height(*strings):
    total_length = sum(len(s) for s in strings)
    # Assuming each character occupies a certain amount of height in the styling
    height_per_character = 0.3  # Adjust this value based on your styling and requirements
    predicted_height = total_length * height_per_character
    return predicted_height+300

classifier = pipeline("sentiment-analysis")
explainer = shap.Explainer(classifier)

option = st.sidebar.radio("Choose an option:", ["Text Sentiment Analysis", "Audio Sentiment Analysis"])

if option == "Text Sentiment Analysis":
    st.title("A Simple Sentiment Analysis Explainer with SHapley Additive exPlanations WebApp.") 

    message = st.text_area("Please Enter your text") 

elif option == "Audio Sentiment Analysis":
    st.title("Audio Sentiment Analysis") 
    st.markdown('[Recording audio here](https://online-voice-recorder.com/)')

    uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "flac"])

    pipe = pipeline("automatic-speech-recognition")

    if uploaded_file is not None:
        data = uploaded_file.read()

    message = pipe(data)['text']

    st.write(message)
    
if st.button("Analyze the Sentiment"): 
    blob = TextBlob(message) 
    result = classifier([message])
    shap_values = explainer([message])

    polarity = result[0]["label"]
    score = result[0]["score"]

    st.success(f"The entered text has {polarity} sentiments associated with it."+str(score)) 
    # st.success(result) 
    # st.success(shap_values)

    st_shap(shap.plots.text(shap_values), height=predict_height(message))

    explanation = shap.Explanation(shap_values)
    id_to_explain = 0 #id to explain
    output_to_explain = 1 #positive or negative

    try:
        st_shap(shap.plots.waterfall(explanation[id_to_explain,:,output_to_explain], max_display=12))
    except Exception as e:
        print(f"An error occurred while generating waterfall plot: {e}")

    try:
        st_shap(shap.plots.bar(explanation[id_to_explain,:,output_to_explain]))
    except Exception as e:
        print(f"An error occurred while generating bar plot: {e}")

    # Given data
    values = shap_values.values
    base_values = shap_values.base_values
    data = shap_values.data

    # Create a dictionary to store the values
    shap_data = {}

    # Iterate through each entry and store SHAP values along with their corresponding sentences
    for i, (shap_values, sentence) in enumerate(zip(values, data)):
        shap_data[i] = {'shap_values': shap_values, 'sentence': sentence}

    # Extracting SHAP values and sentences from the dictionary
    shap_values_0 = shap_data[0]['shap_values']
    sentence_0 = shap_data[0]['sentence']

    # processed the input
    zipped_values = list(zip(shap_values_0, sentence_0))
    sorted_zipped_values = sorted(zipped_values, key=lambda x: x[0][0])
    if len(sorted_zipped_values) >= 10:
        selected_values = sorted_zipped_values[:5] + sorted_zipped_values[-5:]
    else:
        selected_values = sorted_zipped_values

    modified_list = []
    for item in selected_values:
        shap_value_t, word = item
        if shap_value_t[0] < 0 or shap_value_t[1] > 0:
            modified_list.append(('positive', round(shap_value_t[1], 3), word))
        else:
            modified_list.append(('negative', round(shap_value_t[0], 3), word))

    processed_data = [
        {"sentiment": entry[0], "score": entry[1], "word": entry[2]} 
        for entry in modified_list if entry[1] != 0
    ]



    client = OpenAI(
    api_key=os.getenv('OPEN_API_KEY'),
    )


    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "I provide insights based on word sentiment scores:"},
        {"role": "user", "content": f"The overall sentence sentiment is {result}, do backed it up with insightful comments and explaination"},
        {"role": "user", "content": f"{processed_data}"},
        {"role": "user", "content": f"Your input sentence is: {message}."},
    ]
    )

    st.write(completion.choices[0].message.content)
