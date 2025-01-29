# AI-Powered-Disease-Diagnosis-Tool
This project is an AI-powered tool for detecting diseases like **pneumonia** from chest X-ray images. It uses a Convolutional Neural Network (CNN) trained on a dataset of chest X-rays to classify images as either "Normal" or "Pneumonia."

## Features
- **Disease Detection**: Classifies chest X-ray images as "Normal" or "Pneumonia."
- **User-Friendly Interface**: Built with Streamlit for easy interaction.
- **Machine Learning Model**: Uses a CNN model trained on the Chest X-Ray Images (Pneumonia) dataset.

## Technologies Used
- **Python**: Primary programming language.
- **TensorFlow/Keras**: For building and training the CNN model.
- **Streamlit**: For creating the web interface.
- **OpenCV**: For image preprocessing.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For data splitting and evaluation.

## Dataset
The model is trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. The dataset contains chest X-ray images categorized into two classes:
- **Normal**: No signs of pneumonia.
- **Pneumonia**: Signs of pneumonia.

## Installation
Follow these steps to set up the project on your local machine.

### Prerequisites
- Python 3.8 or later
- pip (Python package installer)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/soniprrzz/ai-powered-disease-diagnosis-tool.git
   cd ai-powered-disease-diagnosis-tool
