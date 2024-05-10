import streamlit as st 
from textblob import TextBlob 
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import numpy as np
from streamlit_shap import st_shap
from openai import OpenAI
import os
from dotenv import load_dotenv
import shap
from PIL import Image, ImageOps
import cv2
from io import BytesIO
from deepface import DeepFace
import matplotlib.pyplot as plt

load_dotenv()
shap.initjs()

def predict_height(*strings):
    total_length = sum(len(s) for s in strings)
    # Assuming each character occupies a certain amount of height in the styling
    height_per_character = 0.3  # Adjust this value based on your styling and requirements
    predicted_height = total_length * height_per_character
    return predicted_height+300

def extract_frames(video_path, frames_dir):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return frame_count

def save_uploaded_video(uploaded_file):
    try:
        # Save the uploaded video to a temporary file
        temp_file = BytesIO(uploaded_file.getvalue())
        video_path = "temp_video.mp4"  # Name of the temporary video file
        with open(video_path, "wb") as f:
            f.write(temp_file.read())

        return video_path
    except Exception as e:
        st.error(f"Error saving uploaded video: {e}")
        return None
    
def detect_emotion_with_deepface(frame_path):
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB as expected by deepface
    try:
        # Analyze the image for emotions
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list) and len(analysis) > 0:
            emotions = analysis[0]['emotion']  # Accessing the 'emotion' dictionary from the first element of the list
            print("Emotions:", emotions)
            return emotions
        else:
            print("No emotions detected.")
            return None
    except Exception as e:
        print(f"Error processing {frame_path}: {str(e)}")
        return None

@st.cache_resource 
def load_model():
    return pipeline("sentiment-analysis")

@st.cache_resource 
def load_speech_regcognition_model():
    return pipeline("automatic-speech-recognition")

@st.cache_resource 
def load_img_model():
    return pipeline("image-classification", model="trpakov/vit-face-expression")

@st.cache_resource 
def load_img_processor():
    return AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")

@st.cache_resource 
def load_img_model_classification():
    return AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")


option = st.sidebar.radio("Choose an option:", ["Text Sentiment Analysis", "Audio Sentiment Analysis", "Image Sentiment Analysis", "Video Sentiment Analysis"])

if option == "Text Sentiment Analysis":
    st.title("A Simple Sentiment Analysis Explainer with SHapley Additive exPlanations WebApp.") 

    message = st.text_area("Please Enter your text") 

elif option == "Audio Sentiment Analysis":
    st.title("Audio Sentiment Analysis") 
    st.markdown('[Recording audio here](https://online-voice-recorder.com/)')

    uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "flac"])

    pipe = load_speech_regcognition_model()

    if uploaded_file is not None:
        data = uploaded_file.read()

    message = pipe(data)['text']

    st.write(message)

if option == "Image Sentiment Analysis":
    st.title("Image Sentiment Analysis") 
    st.link_button('Take Image here', 'https://webcamtests.com/take-photo')

    uploaded_file = st.file_uploader("Upload your facial image file", type=["jpg", "jpeg", "png"])

    pipe = load_img_model()

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

    st.image(bytes_data)

elif option == "Video Sentiment Analysis":
    st.title("Video Sentiment Analysis")
    st.link_button('Take video here', 'https://webcamera.io/#google_vignette')

    uploaded_file = st.file_uploader("Upload your video here", type=["mp4"])

    if uploaded_file is not None:
        video_path = save_uploaded_video(uploaded_file)
        if video_path:
            st.success("Video uploaded successfully!")
        else:
            st.error("Failed to upload video.")

        frames_dir = 'extracted_frames'
        total_frames = extract_frames(video_path, frames_dir)
        st.write(f"Extracted {total_frames} frames.")

        # Perform sentiment analysis on each frame using DeepFace
        sentiments = []
        for i in range(total_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i}.jpg")
            result = detect_emotion_with_deepface(frame_path)

            if result:
                dominant_emotion = max(result, key=result.get)
                sentiments.append(dominant_emotion)
            else:
                sentiments.append("No Emotion Detected")

        # Aggregate sentiment over all frames
        sentiment_counts = {emotion: sentiments.count(emotion) for emotion in set(sentiments)}
        message = max(sentiment_counts, key=sentiment_counts.get)
        #st.write(f"Overall sentiment of the video: {overall_sentiment}")
        st.write(message)

if st.button("Analyze the Sentiment"): 
    if (option == "Text Sentiment Analysis" or option == "Audio Sentiment Analysis"):
        classifier = load_model()
        explainer = shap.Explainer(classifier)
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
    else:
        with Image.open(uploaded_file) as image:
            res = pipe(image)
            st.write(res)

            image_np = np.array(image)
            processor = load_img_processor()
            inputs = processor(images=image, return_tensors="pt")
            model = load_img_model_classification()

            def f(img):
                tmp = img.copy()
                inputs = processor(images=tmp, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                return logits

            class_names = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

            # define a masker that is used to mask out partitions of the input image.
            masker = shap.maskers.Image("blur(128,128)", image_np.shape)

            # create an explainer with model and image masker
            explainer = shap.Explainer(f, masker, output_names=class_names)

            # (1, 900, 601, 3)
            reshaped_img = np.expand_dims(image_np, axis=0)

            # here we explain one images using 500 evaluations of the underlying model to estimate the SHAP values
            shap_values = explainer(
                reshaped_img, max_evals=100, batch_size=1, outputs=shap.Explanation.argsort.flip[:4]
            )

            fig = plt.figure()

            shap.image_plot(shap_values, show=False)

            fig = plt.gcf()
            fig.set_size_inches(15, 24)
            st.pyplot(fig)

            del explainer
            del shap_values
            del masker