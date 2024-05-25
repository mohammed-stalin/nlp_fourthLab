
# PART 1 : Arabic Text Preprocessing and Scoring Pipeline

## Overview

This project focuses on preprocessing and scoring Arabic texts for relevance. The pipeline includes steps for cleaning, normalizing, tokenizing, removing stop words, stemming, and scoring the texts based on readability and engagement metrics.

## Project Structure

- **data**: Directory containing input and output data files.
  - `hess_article_content.csv`: Input file containing articles with titles and content.
  - `hess_article_content_processed.csv`: Output file containing processed articles.
- **scripts**: Directory containing Python scripts for various tasks.
  - `preprocess_and_score.py`: Main script for preprocessing text and scoring.

## Dependencies

This project requires the following Python libraries:

- pandas
- BeautifulSoup (bs4)
- nltk
- farasa

You can install the necessary libraries using pip:

```bash
pip install pandas beautifulsoup4 nltk farasa
```

## Setup

### 1. Download Necessary NLTK Data

Before running the scripts, ensure you have the necessary NLTK data:

```python
import nltk
nltk.download('stopwords')
```

### 2. Prepare Input Data

Ensure you have the `hess_article_content.csv` file in the `data` directory. This file should contain the articles with their titles and content.

## Usage

### 1. Preprocess and Score Text

Run the `preprocess_and_score.py` script to preprocess the text and calculate relevance scores:

```bash
python scripts/preprocess_and_score.py
```

This script will:

1. **Read the input CSV file** containing articles.
2. **Handle missing values** by filling NaNs with empty strings.
3. **Clean and normalize the text** by removing HTML tags, non-Arabic characters, and diacritics.
4. **Tokenize, remove stop words, and stem** the text using the Farasa library.
5. **Calculate readability and engagement scores** for each text.
6. **Save the processed text and scores** to a new CSV file.

### 2. Example of Script

Here's the full `preprocess_and_score.py` script:

```python
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer

# Initialize Farasa segmenter and stemmer
farasa_segmenter = FarasaSegmenter(interactive=True)
farasa_stemmer = FarasaStemmer(interactive=True)

# Load Arabic stop words
stop_words = set(stopwords.words('arabic'))

# Define normalization function
def normalize_arabic(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'گ', 'ك', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove Arabic diacritics
    return text

# Function to clean text
def clean_text(text):
    # Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Remove non-Arabic words and numbers using regex
    text = re.sub(r'[^اأإآء-ي\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Define preprocessing function
def preprocess_text(text):
    # Clean text
    text = clean_text(text)
    
    # Normalize text
    text = normalize_arabic(text)
    
    # Tokenize text
    tokens = farasa_segmenter.segment(text).split()
    
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stem words
    stems = [farasa_stemmer.stem(word) for word in tokens]
    
    return ' '.join(stems)

# Define readability and engagement score functions
def calculate_readability(text):
    tokens = text.split()
    num_words = len(tokens)
    num_sentences = text.count('.') + text.count('!') + text.count('؟')  # Sentence ending punctuation in Arabic
    avg_sentence_length = num_words / (num_sentences + 1)  # Avoid division by zero
    readability_score = min(avg_sentence_length / 20, 1) * 10  # Normalized to 10
    return readability_score

def calculate_engagement(text):
    keywords = ['ستالين', 'حرب', 'روسيا']  # Example keywords
    keyword_count = sum(text.count(keyword) for keyword in keywords)
    engagement_score = min(keyword_count / 5, 1) * 10  # Normalized to 10
    return engagement_score

# Path to your existing CSV file
csv_file_path = r'C:\Users\lenovo\Desktop\master\S2\nlp\atelier4\data\hess_article_content.csv'

# Read the CSV file to fetch article content
df = pd.read_csv(csv_file_path)

# Handle missing values by filling NaNs with an empty string
df['text'] = df['text'].fillna('')

# Apply preprocessing to each text in the dataset
df['processed_text'] = df['text'].apply(preprocess_text)

# Calculate scores for each text
df['readability_score'] = df['processed_text'].apply(calculate_readability)
df['engagement_score'] = df['processed_text'].apply(calculate_engagement)
df['total_score'] = (df['readability_score'] + df['engagement_score']) / 2

# Path to save the processed CSV file
processed_csv_file_path = r'C:\Users\lenovo\Desktop\master\S2\nlp\atelier4\data\hess_article_content_processed.csv'

# Save the processed dataset to a CSV file
df.to_csv(processed_csv_file_path, index=False)

print("Preprocessing complete. Processed data saved to CSV.")
```

## Conclusion

This project provides a robust pipeline for preprocessing Arabic text and calculating relevance scores. By following the steps outlined in this README, you can ensure the smooth execution of the pipeline and obtain clean, normalized, and scored text data.

Feel free to extend the pipeline or modify it according to your specific requirements. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.


# PART 2 : Arabic Text Relevance Scoring with Deep Learning Models

This project focuses on predicting relevance scores for Arabic text using various deep learning models. The models used include RNN, Bidirectional RNN, GRU, and LSTM. The relevance scores are normalized using `StandardScaler` during training and denormalized during prediction.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Prediction](#prediction)
- [Models](#models)
- [Preprocessing](#preprocessing)
- [Normalization](#normalization)
- [Saving and Loading Models](#saving-and-loading-models)
- [Denormalizing Scores](#denormalizing-scores)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mohammed-stalin/nlp_fourth_lab.git
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

1. Preprocess your data and prepare the target variable.
2. Train the models using the provided scripts.

### Prediction

1. Use the trained GRU model to predict the relevance score for new Arabic text.

### Example

Here's an example of how to train the models and make predictions:

1. **Training:**
    ```python
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    import pickle

    # Assuming X_train, y_train, X_val, y_val are prepared
    # Normalize the target variable
    scaler = StandardScaler()
    y_train_normalized = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_normalized = scaler.transform(y_val.values.reshape(-1, 1))

    # Define and train your model (example for GRU)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen),
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train_normalized, validation_data=(X_val, y_val_normalized), epochs=10, batch_size=32)

    # Save the model and scaler
    model.save('gru_model.keras')
    with open('scaler.pickle', 'wb') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ```

2. **Prediction:**
    ```python
    import tensorflow as tf
    import pickle
    from sklearn.preprocessing import StandardScaler

    # Load the GRU model
    gru_model = tf.keras.models.load_model('gru_model.keras')

    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Define your new Arabic text
    new_text = "نص عربي جديد للتنبؤ بالنموذج"

    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([new_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)  # Adjust maxlen to the length used during training

    # Predict the score
    predicted_score = gru_model.predict(padded_sequence)

    # Load the scaler object used for target variable normalization during training
    with open('scaler.pickle', 'rb') as handle:
        scaler = pickle.load(handle)

    # Inverse transform the predicted score
    denormalized_score = scaler.inverse_transform(predicted_score)[0][0]

    # Print the denormalized score
    print(f"Denormalized Score: {denormalized_score}")
    ```

## Models

- RNN
- Bidirectional RNN
- GRU
- LSTM

## Preprocessing

Text preprocessing includes tokenization and padding. A tokenizer is created and saved for consistent preprocessing.

## Normalization

Target variable normalization is done using `StandardScaler`. This helps in improving the model performance.

## Saving and Loading Models

Models are saved in the Keras format:
```python
model.save('model_name.keras')
```

The tokenizer and scaler are saved using `pickle`:
```python
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## Denormalizing Scores

To denormalize the predicted scores:
```python
denormalized_score = scaler.inverse_transform(predicted_score)[0][0]
```
# PART 4 : BERT-Based Sentiment Analysis on Amazon Fashion Reviews

## Project Overview
This project leverages a pre-trained BERT model (bert-base-uncased) to perform sentiment analysis on Amazon Fashion reviews. The goal is to fine-tune the BERT model to classify reviews as positive or negative, based on their star ratings. Reviews with 4 or 5 stars are labeled as positive, while reviews with 1, 2, or 3 stars are labeled as negative. The performance of the model is evaluated using standard metrics such as Accuracy, Loss, and F1 Score.

## Dataset
The dataset used in this project is the Amazon Fashion Reviews dataset, which can be downloaded from [here](https://nijianmo.github.io/amazon/index.html). The dataset contains various fields, but we focus primarily on the `reviewText` (the text of the review) and `overall` (the star rating) columns.

## Project Steps

### 1. Load and Preprocess the Data
- Load the dataset from a JSON file.
- Handle missing values in the `reviewText` column by filling them with empty strings.
- Ensure all entries in the `reviewText` column are strings.
- Convert the star ratings into binary labels for sentiment classification: reviews with ratings >= 4 are labeled as positive (1), and the rest are labeled as negative (0).

### 2. Tokenize the Reviews
- Use the `BertTokenizer` from the Hugging Face `transformers` library to tokenize the review texts.
- Pad and truncate the tokenized sequences to a maximum length of 128 tokens.
- Convert the tokenized texts into PyTorch tensors for model input.

### 3. Create DataLoaders
- Split the dataset into training and validation sets.
- Create PyTorch `TensorDataset` and `DataLoader` objects for both training and validation sets to facilitate batch processing.

### 4. Fine-Tune the BERT Model
- Initialize a pre-trained BERT model (`bert-base-uncased`) with a sequence classification head.
- Set up the optimizer (`AdamW`) and learning rate scheduler.
- Train the model for a specified number of epochs, updating the model parameters and learning rate at each step.
- During each epoch, compute the average training loss and validate the model to compute the average validation loss, accuracy, and F1 score.

### 5. Evaluate the Model
- After training, evaluate the model on the validation set using the evaluation function.
- Compute and display the final validation loss, accuracy, and F1 score.

## Model Evaluation Metrics
- **Accuracy**: The proportion of correct predictions out of the total number of predictions.
- **Loss**: The value of the loss function, indicating how well the model is performing.
- **F1 Score**: The harmonic mean of precision and recall, providing a single metric that balances both.

### 6. Conclusion
The pre-trained BERT model is fine-tuned effectively for the sentiment analysis task on the Amazon Fashion reviews dataset. The use of BERT allows leveraging contextual embeddings, which leads to better performance compared to traditional methods. The model's performance is evaluated using standard metrics (Accuracy, Loss, F1 Score), and it demonstrates the effectiveness of pre-trained language models in text classification tasks.



## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
