# AI Workplace Harassment Detection

An AI-based system that detects workplace harassment from written complaints using Natural Language Processing and Transformer models.

---

## Overview

Workplace harassment is a serious issue that often appears in written complaints or reports.  
This project uses a **Transformer-based NLP model** to automatically classify workplace statements into different harassment categories.

The system analyzes text input and predicts whether the statement contains:

• Sexual Harassment  
• Verbal Abuse  
• Physical Threat  
• No Harassment

The application is built with **Python, PyTorch, HuggingFace Transformers, and Streamlit**.

---

## Features

• Detects multiple harassment types from text  
• Transformer-based NLP model for accurate classification  
• Confidence score for predictions  
• Streamlit web interface for interaction  
• Input preprocessing and text cleaning  
• Probability distribution for each class  
• Logging system for predictions

---

## Technologies Used

Python  
PyTorch  
HuggingFace Transformers  
Streamlit  
Pandas  

---

## Project Structure
Workplace_harrasment_detection │ ├── app.py ├── test_model.py ├── requirements.txt ├── tokenizer.json ├── README.md └── .gitignore
---

## Installation

Clone the repository:
git clone https://github.com/sanjana-102005/Workplace_harrasment_detection.git⁠�

Move into the project directory:
cd Workplace_harrasment_detection


Install required packages:
pip install -r requirements.txt


---

## Running the Application

Run the Streamlit application:
streamlit run app.py
The web interface will open in your browser.

---

## Model

The trained model file is **not included in this repository** because it exceeds GitHub's file size limit.

Download the trained model from:After downloading, place the folder inside the project directory as:
harrasment_model/
---

## Example Input

Prediction:
Sexual Harassment Confidence: 94%

---

## Limitations

• Model accuracy depends on training data quality  
• Context understanding is limited  
• May produce false positives or false negatives  

This system should **not be used as the sole basis for HR or legal decisions**.

---

## Author

Sanjana Kulkarni

---

## License

This project is for **educational and research purposes**.
