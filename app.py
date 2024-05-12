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
from moviepy.editor import VideoFileClip
import ffmpeg

load_dotenv()
shap.initjs()

def convert_to_h264(input_video_path, output_video_path):
    ffmpeg.input(input_video_path).output(output_video_path, vcodec='libx264').run()

# TODO: move helper functions out 
def average_sentiment_across_frames(sentiments):
    # Initialize counters for calculating the average sentiment score
    positive_count = 0
    negative_count = 0
    total_score = 0

    # Iterate through each frame's sentiment and score
    for sentiment in sentiments:
        if sentiment["sentiment"] == "positive":
            positive_count += 1
        elif sentiment["sentiment"] == "negative":
            negative_count += 1
        total_score += sentiment["score"]

    # Calculate the total number of frames
    total_frames = len(sentiments)

    # Calculate the average sentiment score
    if total_frames > 0:
        average_score = total_score / total_frames

        # Determine the final sentiment label based on majority
        if positive_count > negative_count:
            final_sentiment = "positive"
        elif negative_count > positive_count:
            final_sentiment = "negative"
        else:
            # If equal, default to neutral or customize your logic
            final_sentiment = "neutral"
    else:
        # Handle edge case if no frames
        final_sentiment = "neutral"
        average_score = 0

    return final_sentiment, average_score

def calculate_weighted_sentiment(data):
    sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}

    # Calculate total score for each sentiment
    for item in data:
        sentiment = item["sentiment"].lower()
        score = item["score"]
        sentiment_scores[sentiment] += score

    # Determine the final sentiment based on the weighted scores
    final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    final_score = sentiment_scores[final_sentiment]

    return final_sentiment, final_score

# map emotions to sentiment
def emotions_to_sentiment(emotions, is_video=False):
    sentiment_mapping = {
        "happy": "positive",
        "angry": "negative",
        "fear": "negative",
        "disgust": "negative",
        "neutral": "neutral",
        "surprise": "positive",
        "sad": "negative"
    }

    converted_results = []

    if is_video:
        for label, score in emotions.items():
            if label in sentiment_mapping:
                sentiment_label = sentiment_mapping[label]
                converted_results.append({
                    "sentiment": sentiment_label,
                    "score": score
                })
            else:
                continue
    else:
        for emotion in emotions:
            label = emotion["label"]
            score = emotion["score"]
            if label in sentiment_mapping:
                sentiment_label = sentiment_mapping[label]
                converted_results.append({
                    "sentiment": sentiment_label,
                    "score": score
                })
            else:
                continue

    return converted_results

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
        frame_path = os.path.join(frames_dir, f"frame_{frame_count+1}.jpg")
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

@st.cache_resource 
def load_img_caption_model(): # BLIP for image captioning
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

@st.cache_resource
def load_face_model():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

is_text = False
is_image = False
is_video = False

option = st.sidebar.radio("Choose an option:", ["Text Sentiment Analysis", "Audio Sentiment Analysis", "Image Sentiment Analysis", "Video Sentiment Analysis"])

if option == "Text Sentiment Analysis":
    is_text = True
    st.title("A Simple Sentiment Analysis Explainer with SHapley Additive exPlanations WebApp.") 

    message = st.text_area("Please enter your text") 

elif option == "Audio Sentiment Analysis":
    st.title("Audio Sentiment Analysis") 
    st.markdown('[Recording audio here](https://online-voice-recorder.com/)')

    uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "flac"])

    pipe = load_speech_regcognition_model()

    if uploaded_file is not None:
        data = uploaded_file.read()

    message = pipe(data)['text']

    st.write(message)

elif option == "Image Sentiment Analysis":
    is_image = True
    st.title("Image Sentiment Analysis") 
    st.link_button('Take Image here', 'https://webcamtests.com/take-photo')

    uploaded_file = st.file_uploader("Upload your facial image file", type=["jpg", "jpeg", "png"])

    pipe = load_img_caption_model()

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        st.header("Image Input with Generated Caption")
        st.image(bytes_data)
        message = pipe(image)

        if message and "generated_text" in message[0]:
                img_caption = message[0]["generated_text"]
                st.write("Image Caption:", img_caption)

elif option == "Video Sentiment Analysis":
    is_video = True
    st.title("Video Sentiment Analysis")
    st.link_button('Take video here', 'https://webcamera.io/#google_vignette')

    uploaded_file = st.file_uploader("Upload your video here", type=["mp4"])

    if uploaded_file is not None:
        video_path = save_uploaded_video(uploaded_file)
        if video_path:
            st.success("Video uploaded successfully!")
            st.video(uploaded_file)
        else:
            st.error("Failed to upload video.")

        video_clip = VideoFileClip(video_path)

        # Extract the audio clip from the video
        audio_clip = video_clip.audio

        # Specify the output audio file path (e.g., "output_audio.wav")
        output_audio_path = "output_audio.wav"

        # Save the audio clip to the specified output file path
        audio_clip.write_audiofile(output_audio_path)

        # Close the video clip and audio clip objects
        video_clip.close()
        audio_clip.close()

        # Load the saved audio file and perform automatic speech recognition (ASR)
        asr_pipe = load_speech_regcognition_model()

        # Read the saved audio file as binary data
        with open(output_audio_path, "rb") as audio_file:
            audio_data = audio_file.read()

        # Perform ASR on the audio data
        vid_caption = asr_pipe(audio_data)['text']

        st.header("Video Input Frames and Speech Recognized")
        # Display the caption obtained from ASR
        st.write("Video Speech:", vid_caption)

        frames_dir = 'extracted_frames'
        total_frames = extract_frames(video_path, frames_dir)
        
        st.write(f"Extracted {total_frames} frames.")

if st.button("Analyze the Sentiment"): 
    # image modality
    if is_image:
      # get emotion labels
      img_pipe = load_img_model()
      emotions = img_pipe(image)

      mapped_sentiment = emotions_to_sentiment(emotions)
      img_sentiment, img_score = calculate_weighted_sentiment(mapped_sentiment)

      message = img_caption

    # video modality
    if is_video:
        # Get emotion-sentiment per frame 
        sentiments = []
        for i in range(1, total_frames-1):
            frame_path = os.path.join(frames_dir, f"frame_{i}.jpg")
            emotions = detect_emotion_with_deepface(frame_path)

            if emotions:
                mapped_sentiment = emotions_to_sentiment(emotions, is_video)
                img_sentiment, img_score = calculate_weighted_sentiment(mapped_sentiment)
                sentiments.append({"frame": i, "sentiment": img_sentiment, "score": img_score})
            else:
                sentiments.append({"frame": i, "sentiment": 'neutral', "score": 0})

        # Aggregate sentiment over all frames
        vid_sentiment, vid_score = average_sentiment_across_frames(sentiments)

        st.success(f"The video has {vid_sentiment.upper()} sentiments associated with it."+str(vid_score)) 

        message = vid_caption

    st.header('Text Sentiment Prediction with SHAP explanation', divider='rainbow')
    # text prediction and explanation
    classifier = load_model()
    explainer = shap.Explainer(classifier)

    blob = TextBlob(message) 
    result = classifier([message])
    shap_values = explainer([message])
    
    polarity = result[0]["label"]
    score = result[0]["score"]

    if is_text:
      st.success(f"The entered text has {polarity} sentiments associated with it."+str(score)) 
    else:
      st.success(f"The text caption has {polarity} sentiments associated with it."+str(score)) 
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

    st.header('LLM explanation on text prediction', divider='rainbow')
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
            {"role": "user", "content": f"The overall sentence sentiment is {result}, do backed it up with insightful comments and explanation"},
            {"role": "user", "content": f"{processed_data}"},
            {"role": "user", "content": f"Your input sentence is: {message}."},
        ]
        )

    st.write(completion.choices[0].message.content)

    if is_image:
        # Provide SHAP explainer on Image modality
        st.header('Image Sentiment Prediction with SHAP explanation', divider='rainbow')
        st.success(f"The image has {img_sentiment.upper()} sentiments associated with it."+str(img_score)) 

        with Image.open(uploaded_file) as image:
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

    # image-text fusion results
    if is_image:
        st.header('Image Multimodal Fusion Results', divider='rainbow')
        img_txt_ret = [
                        {"sentiment": img_sentiment, "score": img_score},
                        {"sentiment": polarity, "score": score}
                    ]
        final_sentiment, final_score = calculate_weighted_sentiment(img_txt_ret)
        st.success(f"The final sentiment has {final_sentiment.upper()} sentiments associated with it."+str(final_score))

        client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))

        completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "I provide insights based on multimodal sentiment fusion results"},
            {"role": "user", "content": f"The overall sentence sentiment is {final_sentiment}, do backed it up with insightful comments and explanation"},
            {"role": "user", "content": f"{img_txt_ret}, first dictionary is image sentiment label and score, second dictionary is text modality."},
            {"role": "user", "content": f"The fusion is based on final_sentiment = max(sentiment_scores, key=sentiment_scores.get)."},
            {"role": "user", "content": f"Your input sentence is: {message}."}
        ])

        st.write(completion.choices[0].message.content)
    
    # provide explanation on video 
    if is_video:
        st.header('Video Sentiment Prediction with Real Time Visualization', divider='rainbow')
        st.success(f"The video has {vid_sentiment.upper()} sentiments associated with it."+str(vid_score)) 
        # <insert stream and visualizations>
        video = cv2.VideoCapture(video_path)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_list = []

        capture = cv2.VideoCapture(video_path)

        for i in range(int(frame_count)):
            _, frame = capture.read()
            face_model = load_face_model()
            face = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        1.1, 5)

            for x, y, width, height in face:
                emotion = DeepFace.analyze(frame, actions = ["emotion"], enforce_detection=False)[0]
                cv2.putText(frame, str(emotion["dominant_emotion"]),
                            (x, y+height),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.9,
                            (255,255,0),
                            2)

                cv2.rectangle(frame, (x, y),
                            (x + width, y + height),
                            (255, 255, 0),
                            2)

                frame_list.append(frame)

            height, width, colors = frame.shape
            size = (width, height)

        output_path = "Emotions.mp4"
        output_246path = "Emotions246.mp4"
        output = cv2.VideoWriter(output_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                20,
                                size)

        for frame in range(len(frame_list)):
            output.write(frame_list[frame])

        output.release()
    
        convert_to_h264(output_path, output_246path)

        st.video(open(output_246path, "rb").read())

    # vid-text fusion results
    if is_video:
        st.header('Video Multimodal Fusion Results', divider='rainbow')
        vid_txt_ret = [
                    {"sentiment": vid_sentiment, "score": vid_score},
                    {"sentiment": polarity, "score": score}
                ]
        final_sentiment, final_score = calculate_weighted_sentiment(vid_txt_ret)
        st.success(f"The final sentiment has {final_sentiment.upper()} sentiments associated with it."+str(final_score)) 

        client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))

        completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "I provide insights based on multimodal sentiment fusion results"},
            {"role": "user", "content": f"The overall sentence sentiment is {final_sentiment}, do backed it up with insightful comments and explanation"},
            {"role": "user", "content": f"{vid_txt_ret}, first dictionary is video sentiment label and score, second dictionary is text modality."},
            {"role": "user", "content": f"The fusion is based on final_sentiment = max(sentiment_scores, key=sentiment_scores.get)."},
            {"role": "user", "content": f"Your input sentence is: {message}."}
        ])

        st.write(completion.choices[0].message.content)
      