# NLP-based-Resume-Screening-App


This project is a **Resume Screening Application** that uses **Natural Language Processing (NLP)** techniques and machine learning to screen resumes based on job descriptions. The system automates the process of shortlisting candidates by matching resumes to job descriptions using various NLP techniques and machine learning models.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [How to Run](#how-to-run)
5. [Data Preprocessing](#data-preprocessing)
   - Text Cleaning
   - Tokenization
   - Vectorization
6. [Model Training](#model-training)
7. [Prediction and Screening](#prediction-and-screening)
8. [App Interface](#app-interface)
9. [Technologies Used](#technologies-used)
10. [Contributing](#contributing)
11. [License](#license)

## Project Overview

The **Resume Screening App** reads through a dataset of resumes and job descriptions, cleans the text, tokenizes it, vectorizes it using techniques like **TF-IDF**, and trains a machine learning model to classify or rank resumes based on how well they match a job description. The user can upload a new resume and get a prediction score based on the job requirements.

## Features

- Text preprocessing (cleaning, tokenization, vectorization)
- Train machine learning model for resume-job matching
- Predict compatibility score between resume and job description
- Simple user interface to upload resumes and job descriptions
- Built with Python and Streamlit for easy deployment

## Installation

To install the necessary dependencies, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/resume-screening-app.git
   cd resume-screening-app
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

To start the app, run the following command:

```bash
streamlit run app.py
```

The web app will open in your default web browser.

## Data Preprocessing

Before training the model, the text in the resumes and job descriptions goes through several preprocessing steps:

### 1. Text Cleaning

We remove unwanted characters, punctuation, and stopwords, as well as convert the text to lowercase.

```python
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
```

### 2. Tokenization

The cleaned text is then tokenized into words.

```python
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def tokenize(text):
    return word_tokenize(text)
```

### 3. Vectorization

We use **TF-IDF Vectorizer** to convert text into numerical features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(resume_texts)
```

## Model Training

We use the vectorized data to train a machine learning model (e.g., Logistic Regression, SVM) that predicts the compatibility of a resume with a job description.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')
```

## Prediction and Screening

After the model is trained, we can use it to predict the compatibility of new resumes with the given job description.

```python
def predict_resume_score(resume_text, job_description):
    resume_vec = tfidf.transform([clean_text(resume_text)])
    job_vec = tfidf.transform([clean_text(job_description)])
    score = model.predict(resume_vec)
    return score
```

## App Interface

We built a simple web interface using **Streamlit** that allows users to upload a resume and job description, and it will display a score based on how well the resume matches the job description.

```python
import streamlit as st

st.title("Resume Screening App")

resume_text = st.text_area("Paste Resume Here")
job_desc = st.text_area("Paste Job Description Here")

if st.button('Predict Match'):
    score = predict_resume_score(resume_text, job_desc)
    st.write(f"Match Score: {score}")
```

## Technologies Used

- Python
- NLP libraries: `nltk`, `re`
- Machine learning: `scikit-learn`
- Web app framework: `Streamlit`
- Vectorization: `TF-IDF`

## Contributing

Feel free to submit a pull request if you would like to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
