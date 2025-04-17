import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Face Similarity Detector",
    page_icon="🧠",
    layout="wide"
)

# Load saved models and components
@st.cache_resource
def load_models():
    try:
        with open("saved_models/svm_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("saved_models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("saved_models/pca.pkl", "rb") as f:
            pca = pickle.load(f)
        with open("saved_models/model_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        return model, scaler, pca, model_info
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, scaler, pca, model_info = load_models()

# Function to preprocess images
def preprocess_image(image):
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Resize to match the training data (62x47)
    img_resized = cv2.resize(img_array, (47, 62))
    
    # Convert to RGB if grayscale
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    return img_normalized

# Function to extract features from two images
def extract_features(img1, img2):
    # Get difference between images
    diff = np.abs(img1 - img2)
    
    # Create flat features
    flat_features = diff.flatten()
    
    # Create engineered features
    eng_features = np.array([
        np.mean(diff),
        np.std(diff),
        np.sum(diff)
    ])
    
    return flat_features, eng_features

# Function to predict similarity
def predict_similarity(img1, img2):
    if model is None or scaler is None or pca is None:
        st.error("Models not loaded correctly.")
        return None
    
    # Preprocess images
    processed_img1 = preprocess_image(img1)
    processed_img2 = preprocess_image(img2)
    
    # Extract features
    flat_features, eng_features = extract_features(processed_img1, processed_img2)
    
    # Scale flat features
    flat_scaled = scaler.transform([flat_features])
    
    # Apply PCA
    pca_features = pca.transform(flat_scaled)
    
    # Combine features
    combined_features = np.hstack([pca_features, eng_features.reshape(1, -1)])
    
    # Make prediction
    similarity_prob = model.predict_proba(combined_features)[0][1]
    
    return similarity_prob

# UI
st.title("🧠 Face Similarity Detector")
st.markdown("""
This app compares two face images and determines if they belong to the same person.
Upload two face images to get started!
""")

# Model information
if model_info:
    with st.expander("📊 Model Information"):
        st.write(f"Model Accuracy: {model_info['accuracy']:.4f}")
        st.write(f"Best Parameters: {model_info['best_params']}")
        st.write("Features used:")
        st.write("- PCA components (128)")
        st.write(f"- Engineered features: {', '.join(model_info['feat_columns'])}")

# Upload images
col1, col2 = st.columns(2)

with col1:
    st.subheader("First Face Image")
    uploaded_file1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"])
    if uploaded_file1:
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption="First Face", use_column_width=True)

with col2:
    st.subheader("Second Face Image")
    uploaded_file2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"])
    if uploaded_file2:
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption="Second Face", use_column_width=True)

# Prediction
if uploaded_file1 and uploaded_file2:
    st.subheader("Similarity Analysis")
    
    if st.button("Compare Faces"):
        with st.spinner("Analyzing face similarity..."):
            similarity = predict_similarity(image1, image2)
            
            if similarity is not None:
                st.subheader(f"Similarity Score: {similarity:.2f}")
                
                # Progress bar for visualization
                st.progress(similarity)
                
                # Interpretation
                if similarity > 0.7:
                    st.success("✅ These faces likely belong to the same person.")
                elif similarity > 0.5:
                    st.warning("⚠️ These faces might belong to the same person.")
                else:
                    st.error("❌ These faces likely belong to different people.")
                
                # Confidence visualization
                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.metric("Same Person", f"{similarity*100:.1f}%")
                with fig_col2:
                    st.metric("Different People", f"{(1-similarity)*100:.1f}%")
            else:
                st.error("Error occurred during prediction.")

# Instructions
with st.expander("📝 How to use"):
    st.markdown("""
    1. Upload two face images using the upload buttons above
    2. Click the "Compare Faces" button
    3. View the similarity score and interpretation
    
    For best results:
    - Use clear, well-lit face images
    - Face should be centered in the image
    - Similar pose and expression will improve accuracy
    """)

# Footer
st.markdown("---")
st.markdown("Face Similarity Detector | Built with Streamlit and scikit-learn")