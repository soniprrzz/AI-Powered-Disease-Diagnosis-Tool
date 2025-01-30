# AI-Powered Disease Diagnosis Tool

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

2. **Create a virtual environment**:
   ```bash
   Copy
   python -m venv venv
   
3. **Activate the virtual environment**:

- **On Windows**:

   ```bash
   .\venv\Scripts\activate
   
- **On macOS/Linux**:

    ```bash
    source venv/bin/activate
    
4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   
5. **Download the dataset**:
   
- **Download the Chest X-Ray Images (Pneumonia) dataset**.

- **Extract the dataset into the data folder. The folder structure should look like this**:
   ```bash
   data/
     ├── train/
     │   ├── NORMAL/
     │   └── PNEUMONIA/
     ├── test/
     │   ├── NORMAL/
     │   └── PNEUMONIA/
     └── val/
         ├── NORMAL/
         └── PNEUMONIA/
   
6. **Train the model**:

   ```bash
   python model.py
  
7. **Run the Streamlit app**:

   ```bash
   streamlit run app.py

### Usage
1. Open the Streamlit app in your browser.

2. Upload a chest X-ray image (in JPG or PNG format).

3. The app will display the prediction: "Normal" or "Pneumonia."

Project Structure
   ```bash
AI-Powered Disease Diagnosis Tool/
├── venv/                     # Virtual environment
├── data/                     # Dataset folder
│   ├── train/                # Training images
│   ├── test/                 # Testing images
│   └── val/                  # Validation images
├── app.py                    # Streamlit app
├── model.py                  # CNN model training script
├── data_preprocessing.py     # Data preprocessing script
├── pneumonia_cnn_model.h5    # Trained CNN model
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
