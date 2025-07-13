# Text Emotion Classification

# 1. Overview

### 1.1 Objectives

- Preprocess raw text data for sentiment analysis.  
- Train a sentiment classifier to classify the data into predefined classes.  
- Evaluate the model by: Accuracy, Precision, Recall, F1-Score and Confusion Matrix.  
- Apply Named Entity Recognition (NER) to extract entity type.  
- Document the outputs. 

### 1.2 Dataset

The dataset used in this task is called “tweet_eval” [1]. TweetEval consists of seven heterogeneous tasks in Twitter, all framed as multi-class tweet classification which make it suitable for the task’s objectives. The dataset has noise that would affect the model result such as punctuation, stopwords, emojis, short words and URLs so it’s a good choice to perform preprocessing methods. 

## 2. Data Observation

### 2.1 Discover the dataset

<img width="1662" height="554" alt="image" src="https://github.com/user-attachments/assets/6d44457f-8314-493d-ab83-f4d27202afad" />

As shown in figure 1, we see that our dataset is divided into three classes, train dataset, test dataset and validation dataset. The train dataset contains 45,615 rows of data (tweets) and is used specifically in sentiment analysis as well as Named Entity Recognition (NER). 

For that we discovered the train dataset to see its structure (figure 2). There are two columns, `text` and `label`—the text is the tweet and the label is the class that indicates whether the tweet is positive, neutral or negative. Positive is number 2, neutral is number 1 and negative is number 0. In figure 3 we can see the first 10 tweets and their labels.

<img width="1432" height="712" alt="image" src="https://github.com/user-attachments/assets/84ad0cd3-ce29-4fe2-97f4-6766519ce547" />

### 2.2 Preprocessing data

<img width="1334" height="786" alt="image" src="https://github.com/user-attachments/assets/862fd71e-56c2-416a-8cf0-e3e65e4b0caa" />

Figure 4 shows two functions to clean the data, `remove_pattern` and `clean_text`, that perform lowercasing all text for consistency, removing specific patterns such as “RT” which means retweet, stripping out URLs, mentions, hashtags, punctuation and numbers, removing emojis and extra space, removing common English stopwords and short words (length ≤ 3). The cleaned output is optimized for tokenization and modeling (figure 5). 

<img width="1215" height="690" alt="image" src="https://github.com/user-attachments/assets/6b4f3033-38b8-4c04-8d8b-9e10e05c72db" />


### 2.3 Perform tokenization

<img width="1267" height="780" alt="image" src="https://github.com/user-attachments/assets/beedb515-8731-41fc-9069-0b4e5091045f" />

As shown in figure 6, we used a pre-trained tokenizer from BERT called `bert-base-uncased` to convert the cleaned tweets into numbers that the model can understand. This process is called tokenization. Each word is mapped to a unique ID from BERT’s vocabulary. We also set a maximum length of 64 tokens to keep all tweets the same size; this is important because BERT needs inputs of equal length. Tweets are usually short, so 64 is more than enough in most cases. The result was added to a new column called “tokenized” and finally the updated dataset was saved for use in modeling and Named Entity Recognition (NER) named “cleaned_tokenized_tweets.csv.” 

### 2.4 Visualizations

<img width="1043" height="660" alt="image" src="https://github.com/user-attachments/assets/5192e7be-10f4-4076-8d61-e4aa290c6c0d" />

<img width="976" height="526" alt="image" src="https://github.com/user-attachments/assets/428d4dba-8c94-470e-aa15-b81e84338a6a" />

As shown in figures 7 & 8, this bar plot displays the distribution of sentiment classes in the dataset, which is Positive, Neutral and Negative. We can clearly see that the Neutral class is the most frequent (45.3%) followed by Positive (39.1%) while the Negative class has the fewest examples (15.5%). Understanding this distribution helps guide preprocessing and evaluation decisions later in the analysis. 



<img width="906" height="579" alt="image" src="https://github.com/user-attachments/assets/bfe9231f-5b44-462f-96eb-37c67fda2cf9" />


In Figure 9, we plot a histogram of the sentence length distribution which shows how many words appear in each tweet after cleaning. Most tweets contain a small number of words typically between 5 and 15 which is expected due to the character limit of tweets. This helps us confirm that the chosen maximum token length (64) is accurate. 



<img width="911" height="500" alt="image" src="https://github.com/user-attachments/assets/cfdfbe43-b0dd-4a7e-943f-1db596b902a7" />

In Figure 10, we plot the average word length per tweet. This gives insight into how complex or simple the word choices are. Most words on average between 5 and 8 characters in length show that the language used is relatively casual and simple which is typical in tweets. Both distributions help us better understand the structure of the text and guide preprocessing and modeling decisions. 




<img width="1091" height="604" alt="image" src="https://github.com/user-attachments/assets/c4b94889-54d7-4897-8c5b-2f88b84e8dd7" />

In Figure 11, we generated a word cloud to visually display the most frequently used words in the cleaned tweets. In a word cloud, words that appear more often are shown in larger and bolder font making it easy to know the common terms at one sight. Here example of the most frequent terms “tomorrow”, “today”, “going” and “time.” 


## 3. Sentiment Classification

### 3.1 Import modules

<img width="1087" height="645" alt="image" src="https://github.com/user-attachments/assets/b9c5d621-1e6e-4104-b01b-01c1c8caa3a5" />

To train our sentiment classification model, we started by converting the cleaned tweets into numbers using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization which helps the model understand the importance of each word, also limiting it to the top 5,000 most relevant words to keep the model focused and efficient and to avoid including rare or unhelpful words. 

Then, we split the data into a training set (80%) and a test set (20%), so we could train the model and test how well it performs. We used a Logistic Regression model which is commonly used for text classification. Also made sure it handled class imbalance by setting `class_weight="balanced"` and increased the number of training iterations to make sure the model had enough time to learn. In the end, the model can predict whether a tweet is positive, neutral or negative. 


### 3.2 Evaluate the model

<img width="1067" height="469" alt="image" src="https://github.com/user-attachments/assets/b6b36541-47a2-43d3-8158-2a4afb6e50e7" />

To evaluate the performance of the classification model we used several key evaluation metrics: Accuracy, Precision, Recall and F1-score. These metrics provide a deeper understanding of how well the model performs across different sentiment classes (Negative, Neutral, Positive). 

Accuracy measures the overall correctness of the model. It was 62.3% which means the model correctly predicted the sentiment in approximately 6 out of 10 tweets. 

F1-score (macro average) was 0.60, indicating a balanced performance between precision and recall across all classes, especially considering class imbalance. From the detailed classification report:  

- The model performed best on Positive tweets (F1 = 0.67), followed by Neutral tweets (F1 = 0.63).  
- It performed less accurately on Negative tweets (F1 = 0.51) possibly due to fewer training examples (has the least support) or ambiguous words.  


Macro Average gives equal weight to all classes and results in:  
- Precision = 0.60  
- Recall = 0.63  
- F1-score = 0.60  

These results suggest the model performs generally well, especially on Neutral and Positive tweets but can be improved further for the Negative class. 



<img width="1065" height="775" alt="image" src="https://github.com/user-attachments/assets/0f817de5-d776-460d-b0cd-6b332dd860dd" />

The confusion matrix (Figure 14) gives a detailed view of the model’s classification performance by comparing the true labels with the predicted labels across the three sentiment classes: Negative, Neutral, and Positive.  

Diagonal values represent correct predictions:  
- 873 Negative tweets were correctly classified as Negative.  
- 2,501 Neutral tweets were correctly classified as Neutral.  
- 2,306 Positive tweets were correctly classified as Positive.  

Off-diagonal values show misclassifications:  
- 342 Negative tweets were incorrectly classified as Neutral.  
- 164 Negative tweets were misclassified as Positive.  
- 832 Neutral tweets were predicted as Negative, while 779 were predicted as Positive.  
- 961 Positive tweets were labeled as Neutral, and 365 were predicted as Negative.  


This matrix highlights:  
- The model is most confident and accurate with Neutral and Positive tweets.  
- It struggles more with Negative tweets, frequently confusing them with Neutral.  
- There’s noticeable confusion between Neutral and Positive, which may indicate overlapping vocabulary or sentiment ambiguity in some tweets.  

The confusion matrix complements the evaluation metrics by visually emphasizing the types and frequency of prediction errors made by the model. 


### 3.3 Model testing

<img width="1023" height="709" alt="image" src="https://github.com/user-attachments/assets/bd893ba4-c474-4173-9d4b-ea19be201dbf" />

Figure 16 shows an output of the prediction function (see figure 15) that uses the trained model to classify the first 10 tweets and map it to its corresponding label, Sentiment Prediction Results: 

- Out of 10 tweets the model correctly predicted the sentiment for 7 tweets.  
- The model performed well on Neutral and Positive tweets, mostly matching the true labels.  
- Some discrepancies occurred where the model predicted Negative for tweets actually labeled Positive (Tweet 1) or predicted Neutral instead of Positive (Tweet 6).  
- Negative tweets were correctly identified (Tweet 8).  
- The cleaned tweet texts show the essential content used for prediction after removing noise.

Overall, the model demonstrates decent accuracy, especially on Neutral and Negative labels but struggles slightly with some Positive examples. :contentReference

<img width="845" height="816" alt="image" src="https://github.com/user-attachments/assets/ea026fc5-43bf-4c76-87e0-20224925c168" />




<img width="852" height="794" alt="image" src="https://github.com/user-attachments/assets/e31ed0ba-4da8-4e14-8959-d7e600630a46" />

Figure 17 shows an additional test of the model. We write a random sentence to test the model’s accuracy out of tweets dataset domain to gain more insights about the model’s performance. 

The sentences used in this test are much clearer and cleaner than real-life tweets which means that to improve the model performance we can perform more preprocessing for the data or use other advanced models.

## 4. Named Entity Recognition (NER)

### 4.1 Apply NER using SpaCy

<img width="700" height="490" alt="image" src="https://github.com/user-attachments/assets/011c8bf6-c298-4fa8-9295-fb0f3d261134" />

In figure 18 we loaded a dataset of pre-cleaned tweets from a CSV file (that we saved in the tokenization phase) and extracted the `clean_text` column and made sure all entries were valid strings. Then initialized SpaCy’s small English language model (`en_core_web_sm`) to prepare for more natural language processing tasks such as named entity recognition (NER). 



<img width="863" height="604" alt="image" src="https://github.com/user-attachments/assets/75c85c23-706b-46a7-8499-bc659228b53b" />

As shown in figure 19, we used a custom function to extract important named entities from the tweets. The function processes each tweet using SpaCy and focuses only on specific types like PERSON, ORG, GPE, DATE and TIME to keep the results meaningful. It returns a dictionary that counts how often each entity appears, along with the processed documents. This helps us better understand the kind of real-world information mentioned in the tweets and gives more insight into the topics or people users are talking about which is a key result in NLP. Also, we can use the `named_entities` and `processed_docs` to perform more analysis and functions. 



<img width="905" height="701" alt="image" src="https://github.com/user-attachments/assets/f8330fb0-fd3f-4057-806e-5aed72f614ba" />

Figures 20 & 21 show an example of how we benefit from the `return_entities_and_processed_docs` function: `print_top_10` outputs the top 10 most frequent entities for each category (e.g., PERSON, ORG, DATE, etc.), sorting them by frequency and printing only those with more than one occurrence. This allows us to quickly identify the most relevant names, organizations, places and times mentioned in the tweets which give a clearer picture of frequent topics and real-world references discussed by users. 




<img width="786" height="542" alt="image" src="https://github.com/user-attachments/assets/b539f7ac-b981-406f-80b8-7e6ccf333db8" />
<img width="793" height="495" alt="image" src="https://github.com/user-attachments/assets/b4d239a6-2f7d-4abf-a0ca-3bac8fd7a16f" />

In figure 22, we used a function to show where a specific named entity appears in the tweets. We pass the name we want to visualize like “john cena” and the type of entity like PERSON. The function goes through the processed tweets and finds the sentences that include this entity, then highlights it using SpaCy’s visual tool `displacy`. This makes it easier to see how the name is used in context and gives us a better understanding of how users talk about certain people or topics. Figure 23 shows another example of the same function. 



<img width="867" height="323" alt="image" src="https://github.com/user-attachments/assets/c125afd2-932d-462e-b770-3cb0344da86e" />

Figure 24 provides an additional example of using SpaCy’s visual tool `displacy` with a random sentence outside the tweet dataset domain. 

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
   git clone https://github.com/Ghaida-232/Task2.git
   cd Task2
   pip install pandas scikit-learn spacy matplotlib wordcloud
   python -m spacy download en_core_web_sm
   ```
2. prepare data
   run notebook `Training _task2.ipynb` or:

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

- Dataset used in this task: https://huggingface.co/datasets/cardiffnlp/tweet_eval 
- Sentiment Classification and Preprocessing notebook: https://colab.research.google.com/drive/1g653PaCDNzmhi9cq_wZ2EVSH3PH507_M?usp=sharing 
- Named Entity Recognition (NER) notebook: https://colab.research.google.com/drive/1F6cRj9ncne72PIg8FHNH1BVljAbtr6f?usp=sharing
