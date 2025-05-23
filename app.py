import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64

# Load model and tokenizer
model = tf.keras.models.load_model("imdb_gru_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200

# Set page config
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")

# Set only background image
def set_bg_image(image_file):
    with open(image_file, "rb") as img:
        b64_img = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("background.jpg")  # Make sure background.jpg is in the same folder

# Custom CSS
st.markdown("""
    <style>
    .title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .imdb {
        background-color: #f5c518;
        color: black;
        padding: 2px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .box {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 30px;
        border-radius: 15px;
        max-width: 700px;
        margin: 40px auto;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .emoji {
        font-size: 60px;
        text-align: center;
        animation: pulse 1.2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# Title with IMDB styled text
st.markdown('<div class="title">🎬 <span class="imdb">IMDB</span> Movie Review Sentiment Classifier</div>', unsafe_allow_html=True)

# UI Box
st.markdown('<div class="box">', unsafe_allow_html=True)

user_input = st.text_area("✍️ Enter a Movie Review:", height=150, placeholder="Example: This movie was a masterpiece...")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]

        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        emoji = "😊" if prediction >= 0.5 else "😞"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        confidence_percent = int(confidence * 100)
        bar_color = "#00cc44" if sentiment == "Positive" else "#cc0000"

        # Display results
        st.markdown(f"<div class='emoji'>{emoji}</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>Sentiment: <span style='color:{bar_color}'>{sentiment}</span></h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Confidence: {confidence_percent}%</p>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background-color:#ddd; border-radius:10px; height:20px; margin-top:10px;">
                <div style="width:{confidence_percent}%; background-color:{bar_color}; height:100%; border-radius:10px; transition: width 0.6s;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)  # Close UI box
