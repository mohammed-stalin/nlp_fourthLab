
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
# PART 3: Fine-Tuning GPT-2 for Quote Generation

## Project Overview
This project demonstrates how to fine-tune the pre-trained GPT-2 language model on a custom dataset of quotes and use it to generate new quotes. The fine-tuning process involves training the model on a dataset of quotes to adapt its language generation capabilities to the style and content of the quotes.

## Steps

### 1. Install Required Libraries
Install the necessary libraries, including `transformers` and `torch`, using the following command:
```bash
!pip install transformers torch
```

### 2. Load the Pre-trained GPT-2 Model and Tokenizer
Load the GPT-2 model and tokenizer from the Hugging Face `transformers` library:
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
```

### 3. Prepare the Dataset
Create a custom `QuotesDataset` class to load and process the quotes dataset:
```python
class QuotesDataset(Dataset):
    def __init__(self, file_path, tokenizer, end_of_text_token=""):
        super().__init__()
        self.quote_list = []
        self.end_of_text_token = end_of_text_token
        self.tokenizer = tokenizer

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                quote_str = f"{line.strip()} {self.end_of_text_token}"
                self.quote_list.append(quote_str)

    def __len__(self):
        return len(self.quote_list)

    def __getitem__(self, item):
        return self.quote_list[item]
```

### 4. Load the Dataset
Specify the paths to your dataset files and load them:
```python
train_file_path = 'train2.txt'
valid_file_path = 'valid2.txt'
test_file_path = 'test2.txt'

train_dataset = QuotesDataset(train_file_path, tokenizer)
valid_dataset = QuotesDataset(valid_file_path, tokenizer)
test_dataset = QuotesDataset(test_file_path, tokenizer)
```

### 5. Create DataLoaders
Create DataLoaders for batching the data during training:
```python
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [torch.tensor(tokenizer.encode(item)) for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return input_ids_padded

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate_fn)
```

### 6. Fine-Tune the Model
Fine-tune the GPT-2 model on the custom dataset:
```python
from transformers import AdamW, get_linear_schedule_with_warmup

EPOCHS = 20
LEARNING_RATE = 3e-5
WARMUP_STEPS = 500
BATCH_SIZE = 2
MAX_SEQ_LEN = 600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

for epoch in range(EPOCHS):
    model.train()
    for idx, quotes in enumerate(train_dataloader):
        quote_tens = quotes.to(device)
        if quote_tens.size()[1] > MAX_SEQ_LEN:
            continue
        outputs = model(quote_tens, labels=quote_tens)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        model.zero_grad()
    torch.save(model.state_dict(), f"trained_models/gpt2_quotes_{epoch}.pt")
```

### 7. Evaluate the Model
Evaluate the model on the validation set:
```python
model.eval()
validation_loss = 0.0
for idx, quotes in enumerate(valid_dataloader):
    quote_tens = quotes.to(device)
    with torch.no_grad():
        outputs = model(quote_tens, labels=quote_tens)
        loss = outputs.loss
        validation_loss += loss.item()
print(f"Validation loss: {validation_loss / len(valid_dataloader)}")
```

### 8. Generate Quotes
Use the fine-tuned model to generate new quotes:
```python
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)
    chosen_index = np.random.choice(n, 1, p=top_prob)
    token_id = ind[chosen_index][0]
    return int(token_id)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.load_state_dict(torch.load(f"trained_models/gpt2_quotes_{MODEL_EPOCH}.pt"))
model.to(device)

with torch.no_grad():
    for quote_idx in range(100):
        cur_ids = torch.tensor(tokenizer.encode("QUOTE:")).unsqueeze(0).to(device)
        for i in range(100):
            outputs = model(cur_ids)
            logits = outputs.logits
            softmax_logits = torch.softmax(logits[0, -1], dim=0)
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=3)
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)
            if next_token_id == tokenizer.eos_token_id:
                break
        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        with open(f'generated_quotes_{MODEL_EPOCH}.txt', 'a') as f:
            f.write(f"{output_text}\n\n")
print(f"Generated quotes saved to generated_quotes_{MODEL_EPOCH}.txt")
```

# PART 4: BERT-Based Sentiment Analysis on Amazon Fashion Reviews

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
- **Loss**: The value of the loss function, indicates how well the model is performing.
- **F1 Score**: The harmonic mean of precision and recall, providing a single metric that balances both.



## General Conclusion and Learnings

### PART 1: Arabic Text Preprocessing and Scoring Pipeline

**Conclusion:**

The Arabic text preprocessing and scoring pipeline project provided a comprehensive approach to handle and analyze Arabic text. The pipeline successfully managed various preprocessing steps including cleaning, normalizing, tokenizing, removing stop words, and stemming Arabic text. The project also incorporated scoring mechanisms to assess the readability and engagement of the text.

**Learnings:**

1. **Text Normalization and Cleaning:**
   - Learned techniques for cleaning Arabic text, such as removing HTML tags and non-Arabic characters.
   - Understood the importance of normalizing text by addressing variations in Arabic characters and removing diacritics.

2. **Tokenization and Stop Word Removal:**
   - Implemented tokenization and stop word removal using Farasa, a tool specialized for Arabic text processing.
   - Recognized the challenges in handling Arabic stop words and the impact of tokenization on text analysis.

3. **Scoring Mechanisms:**
   - Developed methods to calculate readability and engagement scores.
   - Gained insights into how to create custom scoring functions based on sentence length and keyword frequency.

### PART 2: Arabic Text Relevance Scoring with Deep Learning Models

**Conclusion:**

This project focused on predicting relevance scores for Arabic text using various deep learning models such as RNN, Bidirectional RNN, GRU, and LSTM. The project highlighted the entire workflow from data preprocessing, model training, to prediction and denormalization of scores.

**Learnings:**

1. **Deep Learning Models for Text Analysis:**
   - Learned how to set up and train different RNN-based models (RNN, Bidirectional RNN, GRU, LSTM) for text relevance scoring.
   - Understood the importance of choosing the right model architecture for specific text-processing tasks.

2. **Normalization Techniques:**
   - Used `StandardScaler` for normalizing target variables, improving model performance.
   - Learned how to denormalize predictions to interpret results effectively.

3. **Model Evaluation and Saving:**
   - Gained experience in evaluating model performance using metrics such as mean squared error.
   - Learned best practices for saving and loading models and preprocessing artifacts (tokenizers and scalers).

### PART 3: Fine-Tuning GPT-2 for Quote Generation

**Conclusion:**

The project demonstrated the process of fine-tuning a pre-trained GPT-2 model on a custom dataset of quotes to generate new quotes. This involved preparing a dataset, training the model, and evaluating its performance.

**Learnings:**

1. **Fine-Tuning Pre-trained Models:**
   - Understood the process of fine-tuning GPT-2, including preparing the dataset and configuring the training loop.
   - Learned how to adjust model parameters and use schedulers to optimize training.

2. **Custom Dataset Handling:**
   - Developed skills in creating custom datasets and data loaders for efficient batch processing during training.
   - Recognized the importance of proper data preparation for successful model training.

3. **Quote Generation:**
   - Learned techniques for generating text using a fine-tuned language model.
   - Experimented with controlling the generation process by sampling from the model's output probabilities.

### PART 4: BERT-Based Sentiment Analysis on Amazon Fashion Reviews

**Conclusion:**

This project focused on fine-tuning a BERT model for sentiment analysis on Amazon Fashion reviews. The model classified reviews as positive or negative based on their star ratings, demonstrating the effectiveness of transfer learning in text classification tasks.

**Learnings:**

1. **Leveraging BERT for Text Classification:**
   - Gained experience in fine-tuning BERT for a specific classification task.
   - Understood the benefits of using pre-trained models for text analysis, especially in capturing contextual information.

2. **Data Preprocessing and Tokenization:**
   - Learned to preprocess text data, including handling missing values and tokenizing text using `BertTokenizer`.
   - Managed sequence lengths through padding and truncation to fit the model’s input requirements.

3. **Model Training and Evaluation:**
   - Developed skills in setting up the training loop, including configuring the optimizer and learning rate scheduler.
   - Evaluated model performance using metrics such as Accuracy, Loss, and F1 Score, gaining insights into model effectiveness.

### Overall Learnings:

- **Integration of Various NLP Tools:** Gained practical experience in integrating various NLP tools and libraries, including NLTK, Farasa, TensorFlow, and Hugging Face's Transformers.
- **Deep Learning Techniques:** Improved understanding of deep learning techniques and architectures suitable for text analysis tasks.
- **End-to-End Project Management:** Developed skills in managing end-to-end machine learning projects, from data preprocessing and model training to evaluation and deployment.
- **Handling Arabic Text:** Enhanced knowledge of handling and processing Arabic text, addressing its unique challenges in NLP tasks.
- **Transfer Learning Benefits:** Recognized the advantages of transfer learning and fine-tuning pre-trained models for specialized tasks, improving performance and efficiency.

 ## License

This project is licensed under the MIT License. See the [include](File:LICENCE) file for more details.

---
