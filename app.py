import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import cv2
import tempfile
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Ensure the required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Cache the model and tokenizer to avoid downloading them repeatedly
@st.cache_resource
def load_model_and_tokenizer():
    model_id = "vikhyatk/moondream2"
    revision = "2024-07-23"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision,
        torch_dtype=torch.float16).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

# Function to extract frames from video and their timestamps
def extract_frames_with_timestamps(video_path, interval=0.2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, image = cap.read()
    count = 0
    
    while success:
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = timestamp_ms / 1000.0
        
        if count % (interval * frame_rate) == 0:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(img)
            timestamps.append(timestamp_sec)
        
        success, image = cap.read()
        count += 1
    
    cap.release()
    
    print(f"Total frames captured: {len(frames)}")
    return frames, timestamps

# Function to filter and clean description
def filter_description(description):
    words = nltk.word_tokenize(description)
    tagged_words = nltk.pos_tag(words)
    
    # Filter only nouns (NN, NNS), pronouns (PRP, PRP$), and adjectives (JJ, JJR, JJS)
    filtered_words = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'PRP', 'PRP$', 'JJ', 'JJR', 'JJS']]
    
    return filtered_words

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Streamlit UI
st.title("CCTV Video Analyzer")
st.write("Upload CCTV footage to get descriptions of the video frames.")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])


if uploaded_video is not None:
    # Save the video to the current directory
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    # Extract frames and timestamps from the video
    frames, timestamps = extract_frames_with_timestamps(video_path, interval=1)  # Extract 1 frame per second
    
    # Process each frame using the model
    descriptions = []
    prev_description_words = set()
    key_frames = []

    with st.spinner("Processing..."):
        for i, frame in enumerate(frames):
            enc_image = model.encode_image(frame)
            description = model.answer_question(enc_image, "Describe this image.", tokenizer)
            filtered_words = list(filter_description(description))  # Convert to list
            new_words = set(filtered_words) - prev_description_words
            if len(new_words) > 5:
                key_frames.append((timestamps[i], frame))
            
            descriptions.append((timestamps[i], filtered_words))
            prev_description_words = set(filtered_words)  # Ensure it remains a set

    # Prepare data for CSV
    max_len = max(len(words) for _, words in descriptions)
    csv_data = []
    for timestamp, words in descriptions:
        row = [timestamp] + words + [''] * (max_len - len(words))  # Fill remaining columns with empty strings
        csv_data.append(row)

    # Display the frames in a grid layout
    num_columns = 3  # Number of columns in the grid
    num_rows = (len(frames) + num_columns - 1) // num_columns  # Calculate number of rows needed
    
    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col in range(num_columns):
            index = row * num_columns + col
            if index < len(frames):
                frame = frames[index]
                cols[col].image(frame, caption=f"Frame {index + 1} at {timestamps[index]:.2f}s")

     # Display key frames in a grid layout
    if key_frames:
        st.write("Key Frames:")
        num_columns_key_frames = 3  # Number of columns for key frames grid
        num_rows_key_frames = (len(key_frames) + num_columns_key_frames - 1) // num_columns_key_frames  # Calculate number of rows needed

        for row in range(num_rows_key_frames):
            cols = st.columns(num_columns_key_frames)
            for col in range(num_columns_key_frames):
                index = row * num_columns_key_frames + col
                if index < len(key_frames):
                    timestamp, frame = key_frames[index]
                    cols[col].image(frame, caption=f"Key Frame {index + 1} at {timestamp:.2f}s")
    
    # Prepare data for CSV
    max_len = max(len(words) for _, words in descriptions)
    csv_data = []
    for timestamp, words in descriptions:
        row = [timestamp] + words + [''] * (max_len - len(words))  # Fill remaining columns with empty strings
        csv_data.append(row)
    
    columns = ['Timestamp'] + [f'Word_{i+1}' for i in range(max_len)]
    csv_df = pd.DataFrame(csv_data, columns=columns)
    
    # Save to CSV
    csv_file = "video_descriptions.csv"
    csv_df.to_csv(csv_file, index=False)
    
    # Display CSV and download option
    st.write("CSV of descriptions:")
    st.dataframe(csv_df)
    st.download_button(label="Download CSV", data=csv_file, file_name=csv_file, mime='text/csv')
    # Remove temporary file
    os.unlink(video_path)