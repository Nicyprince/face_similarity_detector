# 🧠 Face Similarity Detector

A face comparison application that determines if two images contain the same person. This project leverages advanced feature engineering, SVM models, and is deployed using Streamlit.

## 📋 Overview

This application uses machine learning to determine if two face images belong to the same person. The SVM model was trained on the LFW (Labeled Faces in the Wild) dataset and achieved 81.6% accuracy after cleaning and feature engineering.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Nicyprince/face_similarity_detector.git
   cd face-similarity-detector
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

If you need to train the model from scratch:

```bash
python model_training.py
```

This will generate all necessary model files in the `saved_models` directory.

### Running the App

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The app will be available at http://localhost:8501

## 📊 How It Works

1. **Feature Engineering**: The model extracts both:
   - PCA components from the face difference map
   - Statistical features (mean, standard deviation, sum) from the difference image

2. **Preprocessing**: 
   - Images are resized to 62x47 pixels
   - Pixel values are normalized to [0,1]
   - Features are standardized using StandardScaler

3. **Prediction**:
   - The SVM model with RBF kernel predicts the probability that two faces belong to the same person

## 📷 Usage

1. Upload two face images using the provided interface
2. Click "Compare Faces" to analyze
3. View the similarity score and interpretation of results

## 📁 Project Files

- `model_training.py`: Script to train and save the model files
- `app.py`: Streamlit application code
- `requirements.txt`: Required Python packages
- `saved_models/`: Directory containing:
  - `svm_model.pkl`: Trained SVM classifier
  - `scaler.pkl`: StandardScaler for feature normalization
  - `pca.pkl`: PCA model for dimensionality reduction
  - `model_info.pkl`: Configuration and metadata

## 📈 Model Performance

| Model | Without Cleaning/Features | With Cleaning/Features |
|-------|---------------------------|------------------------|
| SVM   | 78.9%                     | 81.6%                  |

Feature engineering and data cleaning significantly improved the model's performance.

