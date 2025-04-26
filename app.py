import streamlit as st
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Class labels
class_labels = {
    0: "Biodegradable",
    1: "Non-Biodegradable"
}

descriptions = {
    "Biodegradable": "🌱 Biodegradable waste consists of organic materials that decompose naturally. Examples include food scraps, leaves, paper, wood and dead animals. These materials can be composted to create nutrient-rich soil, reducing landfill waste and promoting sustainability.",
    "Non-Biodegradable": "🛑 Non-biodegradable waste consists of materials that do not break down naturally, such as plastics, metals, and electronic waste. They continue to stay on earth for thousands of years without any degradation. Proper disposal and recycling help reduce pollution and prevent long-term environmental damage."
}

def preprocess_image(img):
    img = img.resize((128, 128))                    # Resize image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    img_array /= 255.0                              # Normalize pixel values
    return img_array

def predict_waste(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0]    # Get prediction probabilities
    predicted_class = np.argmax(prediction)         # Get the class with highest probability
    confidence = prediction[predicted_class] * 100  # Confidence percentage
    waste_type = class_labels[predicted_class]
    return waste_type, confidence

# CSS for styling

st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #2193b0, #6dd5ed);
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
        }
        .upload-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
        }
        .waste-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.2);
            margin-top: 20px;
            font-family: 'Arial', sans-serif;
            text-align: center;
        }
        .waste-title {
            font-size: 25px;
            font-weight: bold;
            color: #4CAF50;
        }
        .waste-text {
            font-size: 18px;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI

st.markdown("<h1 class='main-title'>🗑️ Waste Classification Web App</h1>", unsafe_allow_html=True)
st.write("Upload an image to classify it as **Biodegradable** or **Non-Biodegradable**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")

if uploaded_file is not None:
    col1, col2 = st.columns(2)  # Create two columns
    
    with col1:
        img = Image.open(uploaded_file)
        img_resized = img.resize((150, 150))
        st.image(img_resized, caption="Uploaded Image", width=250)
        
        if st.button("Classify"):
          with st.status("🔍 Classifying... Please wait", expanded=False) as status:
            waste_type, confidence = predict_waste(img)
            status.update(label="Classification Complete!", state="complete", expanded=False)
          
          st.success(f"🚀 Predicted Waste Type: {waste_type}")
          st.write(f"📊 **Confidence Level:** {confidence:.2f}%")


          with col2:
             if 'waste_type' in locals():  # Display info only after classification
                bg_color = "#D4EDDA" if waste_type == "Biodegradable" else "#F8D7DA"
                text_color = "#155724" if waste_type == "Biodegradable" else "#721C24"
                
                st.markdown(
                    f"""
                    <div style="
                        padding: 15px;
                        border-radius: 10px;
                        background-color: {bg_color};
                        color: {text_color};
                        font-weight: bold;
                        text-align: center;">
                        <h3>About {waste_type} Waste</h3>
                        <p>{descriptions[waste_type]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

