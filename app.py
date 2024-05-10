import streamlit as st 
from textblob import TextBlob 
from transformers import pipeline
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
from moviepy.editor import VideoFileClip

load_dotenv()
shap.initjs()

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

        # Create the final sentiment output (label, score)
        final_output = {"label": final_sentiment, "score": round(average_score, 2)}
    else:
        final_output = {"label": "unknown", "score": 0.0}  # Handle edge case if no frames

    return final_output

def average_sentiment_scores(converted_results):
    # Initialize counters for positive and negative scores
    positive_sum = 0
    positive_count = 0
    negative_sum = 0
    negative_count = 0

    # Iterate over converted results to calculate sums and counts
    for result in converted_results:
        sentiment = result["sentiment"].lower()
        score = result["score"]
        
        if sentiment == "positive":
            positive_sum += score
            positive_count += 1
        elif sentiment == "negative":
            negative_sum += score
            negative_count += 1

    # Calculate average scores for positive and negative sentiments
    if positive_count > 0:
        average_positive_score = positive_sum / positive_count
    else:
        average_positive_score = 0  # Default to 0 if no positive scores
        
    if negative_count > 0:
        average_negative_score = negative_sum / negative_count
    else:
        average_negative_score = 0  # Default to 0 if no negative scores

    # Determine final sentiment prediction based on average scores
    if average_positive_score > average_negative_score:
        final_sentiment = "positive"
        final_score = average_positive_score
    elif average_positive_score < average_negative_score:
        final_sentiment = "negative"
        final_score = average_negative_score
    else:
        final_sentiment = "neutral"  # Handle case where scores are equal
        final_score = average_positive_score  # Use either score as the final score

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

classifier = load_model()
explainer = shap.Explainer(classifier)
is_text = False
is_image = False
is_video = False

option = st.sidebar.radio("Choose an option:", ["Text Sentiment Analysis", "Audio Sentiment Analysis", "Image Sentiment Analysis", "Video Sentiment Analysis"])

if option == "Text Sentiment Analysis":
    is_text = True
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

elif option == "Image Sentiment Analysis":
    is_image = True
    st.title("Image Sentiment Analysis") 
    st.link_button('Take Image here', 'https://webcamtests.com/take-photo')

    uploaded_file = st.file_uploader("Upload your facial image file", type=["jpg", "jpeg", "png"])

    # BLIP for image captioning
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(uploaded_file)

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
        asr_pipe = pipeline("automatic-speech-recognition")

        # Read the saved audio file as binary data
        with open(output_audio_path, "rb") as audio_file:
            audio_data = audio_file.read()

        # Perform ASR on the audio data
        vid_caption = asr_pipe(audio_data)['text']

        # Display the caption obtained from ASR
        st.write("Video Caption (ASR):", vid_caption)

        frames_dir = 'extracted_frames'
        total_frames = extract_frames(video_path, frames_dir)
        st.write(f"Extracted {total_frames} frames.")

if st.button("Analyze the Sentiment"): 
    # image modality
    if is_image:
      # get emotion labels
      img_pipe = pipeline("image-classification", model="trpakov/vit-face-expression")
      st.image(bytes_data)
      emotions = img_pipe(image)

      mapped_sentiment = emotions_to_sentiment(emotions)
      img_sentiment, img_score = average_sentiment_scores(mapped_sentiment)
      st.write(img_sentiment, img_score)

      st.success(f"The image has {img_sentiment.upper()} sentiments associated with it."+str(img_score)) 

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
                img_sentiment, img_score = average_sentiment_scores(mapped_sentiment)
                sentiments.append({"frame": i, "sentiment": img_sentiment, "score": img_score})
            else:
                sentiments.append({"frame": i, "sentiment": 'neutral', "score": 0})

        # Aggregate sentiment over all frames
        vid_sentiment, vid_score = average_sentiment_across_frames(sentiments)
        st.success(f"The video has {vid_sentiment.upper()} sentiments associated with it."+str(vid_score)) 

        message = vid_caption

    # text prediction and explanation
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

    # # OpenAI GPT explanation
    # client = OpenAI(
    # api_key=os.getenv('OPEN_API_KEY'),
    # )


    # completion = client.chat.completions.create(
    # model="gpt-3.5-turbo-0125",
    # messages=[
    #     {"role": "system", "content": "I provide insights based on word sentiment scores:"},
    #     {"role": "user", "content": f"The overall sentence sentiment is {result}, do backed it up with insightful comments and explaination"},
    #     {"role": "user", "content": f"{processed_data}"},
    #     {"role": "user", "content": f"Your input sentence is: {message}."},
    # ]
    # )

    # st.write(completion.choices[0].message.content)

    # image-text fusion results
    if is_image:
      img_txt_ret = [
                    {"sentiment": img_sentiment, "score": img_score},
                    {"sentiment": polarity, "score": score}
                ]
      final_sentiment, final_score = average_sentiment_scores(img_txt_ret)
      st.success(f"The final sentiment has {final_sentiment.upper()} sentiments associated with it."+str(final_score)) 
    
    # vid-text fusion results
    if is_video:
      vid_txt_ret = [
                    {"sentiment": vid_sentiment, "score": vid_score},
                    {"sentiment": polarity, "score": score}
                ]
      final_sentiment, final_score = average_sentiment_scores(vid_txt_ret)
      st.success(f"The final sentiment has {final_sentiment.upper()} sentiments associated with it."+str(final_score)) 
      