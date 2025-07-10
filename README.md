# Text Emotion Classification

## Original Report

You can find the full Task 2 report in PDF format here:

[Download the original report (Task2\_Report.pdf)](Task2%20Report.pdf)

---

## Objective

* preprocess raw tweet text for sentiment analysis
* train a classifier to label tweets as negative, neutral, or positive
* evaluate using accuracy, precision, recall, F1-score, and confusion matrix
* apply named entity recognition (NER) to extract PERSON, ORG, GPE, DATE, TIME

## Dataset Description

The dataset used is **tweet\_eval** \[1], a multi-task Twitter corpus. The sentiment split contains **45,615** training tweets across three classes (negative=0, neutral=1, positive=2), plus validation and test sets. It includes noise like punctuation, stopwords, emojis, and URLs, making preprocessing essential.

## Text Preprocessing

* remove RT tags, URLs, mentions, hashtags, punctuation, numbers
* lowercase all text and strip extra whitespace
* remove emojis, stopwords, and short words (≤ 3 characters)
* output saved as `cleaned_tokenized_tweets.csv` for modeling

## Exploratory Data Analysis (EDA)

* class distribution
  ![Class Distribution](images/class_distribution.png)
  shows neutral 45.3%, positive 39.1%, negative 15.5%

* sentence length distribution
  ![Sentence Length Distribution](images/sentence_length_distribution.png)
  most tweets have 5–15 words

* average word length
  ![Avg Word Length](images/avg_word_length.png)
  typical words are 5–8 characters

* word cloud
  ![Word Cloud](images/wordcloud.png)
  highlights common terms: tomorrow, today, going, time

## Model Training

* vectorize text with TF-IDF (top 5,000 features)
* split data 80% train / 20% test
* train Logistic Regression (class\_weight="balanced", max\_iter=1000)

## Evaluation Metrics

* classification report
  ![Classification Report](images/classification_report.png)
  accuracy 62.3%, macro F1 0.60

* confusion matrix
  ![Confusion Matrix](images/confusion_matrix.png)
  highlights true vs predicted counts per class

## Named Entity Recognition (NER)

* apply SpaCy `en_core_web_sm`
  ![NER Entity Counts](images/ner_entity_counts.png)
  tracks frequency of PERSON, ORG, GPE, DATE, TIME

* context visualization
  ![Entity Context](images/entity_context.png)
  shows how entities appear in tweet sentences

## Project Structure

```text
.
├── data/
│   └── tweet_eval/                # original and cleaned CSVs
├── notebooks/
│   └── Task2_Text_Emotion.ipynb   # preprocessing, EDA, modeling, NER
├── Task2_Report.pdf               # original report document
└── README_Task2.md                # this file
```

## How to Run

1. clone repo and install dependencies

   ```bash
   git clone https://github.com/USERNAME/REPO_NAME.git
   cd REPO_NAME
   pip install pandas scikit-learn spacy matplotlib wordcloud
   python -m spacy download en_core_web_sm
   ```
2. prepare data
   run notebook `Task2_Text_Emotion.ipynb` or:

   ```bash
   python preprocess.py --input data/tweet_eval/raw.csv --output data/cleaned.csv
   ```
3. train & evaluate

   ```bash
   python train_sentiment.py --data data/cleaned.csv
   ```
4. perform NER

   ```bash
   python run_ner.py --data data/cleaned.csv
   ```

## References

1. Cardiff NLP TweetEval. HuggingFace. [https://huggingface.co/datasets/cardiffnlp/tweet\_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
2. Sentiment & NER notebook: [https://colab.research.google.com/drive/1g653PaCDNzmhi9cq\_wZ2EVSH3PH507\_M?usp=sharing](https://colab.research.google.com/drive/1g653PaCDNzmhi9cq_wZ2EVSH3PH507_M?usp=sharing)
