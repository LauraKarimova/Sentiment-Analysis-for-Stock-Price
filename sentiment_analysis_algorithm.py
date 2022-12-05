# Import the libraries necessary for the algorithm to work.  I will mainly use nltk, requests and pandas libraries.
import requests
import json
import numpy as np
import pandas as pd
import nltk 
import matplotlib
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import re
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake 
from datetime import date, timedelta


# Initialize API to create a request directly from the browser.
url = 'https://newsapi.org/v2/everything?'
api_key = 'c1a64fea148341ed92239270ee28c94c' 


# Create function to take raw data from the API and process it into a list in order to transform it into a pandas dataframe.
def get_articles(file): 
    article_results = [] 
    for i in range(len(file)):
        article_dict = {}
        article_dict['title'] = file[i]['title']
        article_dict['author'] = file[i]['author']
        article_dict['source'] = file[i]['source']
        article_dict['description'] = file[i]['description']
        article_dict['content'] = file[i]['content']
        article_dict['pub_date'] = file[i]['publishedAt']
        article_dict['url'] = file[i]["url"]
        article_dict['photo_url'] = file[i]['urlToImage']
        article_results.append(article_dict)
    return article_results


# Collect the first 100 articles to check the API and the working of the function.
from_date = date.today() - timedelta(days=30)

parameters_headlines = {
    'q': 'AAPL',
    'sortBy':'popularity',
    'pageSize': 100,
    'apiKey': api_key,
    'language': 'en',
    'from' : from_date
}
# Make the API call 
response_headline = requests.get(url, params = parameters_headlines)
response_json_headline = response_headline.json()

responses = response_json_headline["articles"]
# Transform the data from JSON dictionary to a pandas data frame.
news_articles_df = pd.DataFrame(get_articles(responses))
# Print the head to check the format and the working of the get_articles function.
news_articles_df.head()
len(news_articles_df)


# Create a list with various new sources and then make a request to get news from the given list of websites.
responses = list() 
domains = ['wsj.com', 'finviz.com','nyse.com','bbc.co.uk','techcrunch.com', 'nytimes.com','bloomberg.com','businessinsider.com',
             'cbc.ca','cnbc.com','cnn.com','ew.com','espn.go.com','espncricinfo.com','foxnews.com', 'apnews.com',
             'news.nationalgeographic.com','nymag.com','cnbc.com', 'reuters.com','rte.ie','thehindu.com','huffingtonpost.com',
             'irishtimes.com','timesofindia.indiatimes.com','washingtonpost.com','time.com','medicalnewstoday.com',
             'ndtv.com','theguardian.com','dailymail.co.uk','firstpost.com','thejournal.ie', 'hindustantimes.com',
             'economist.com','news.vice.com','usatoday.com','telegraph.co.uk','metro.co.uk','mirror.co.uk','news.google.com']
for domain in domains:
    parameters_headlines = {
    'domains':format(domain),
    'sortBy':'popularity',
    'pageSize': 100,
    'apiKey': api_key,
    'language': 'en',
    'from' : from_date    
    }
    rr = requests.get(url, params = parameters_headlines)
    data = rr.json()
    print(data)
    responses = data["articles"]
    news_articles_df=news_articles_df.append(pd.DataFrame(get_articles(responses)))


# Print the head to check the format and the working of the get_articles function.
print(news_articles_df.shape)
news_articles_df.head()


# Create function to extract just the name of the source of the news article and exclude other details.
def source_getter(df):
    source = []
    for source_dict in df['source']:
        source.append(source_dict['name'])
    df['source'] = source 

source_getter(news_articles_df)


# Convertation the publication date to the format for future analysis.
news_articles_df['pub_date'] = pd.to_datetime(news_articles_df['pub_date']).apply(lambda x: x.date())


# Check if the missing data exist
news_articles_df.isnull().sum() 


# Drop the rows with missing data if they exist.
news_articles_df.dropna(inplace=True)
news_articles_df = news_articles_df[~news_articles_df['description'].isnull()]


# Summarize the number of rows and columns in the clear dataset.
print(news_articles_df.isnull().sum())
print(news_articles_df.shape)


# Combine the title and the content to get one dataframe column
news_articles_df['combined_text'] = news_articles_df['title'].map(str) +" "+ news_articles_df['content'].map(str) 
news_articles_df.head()


# Create function to remove non-ascii characters from the text
def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)


# Create function to remove the punctuations, apostrophe, special characters using regular expressions.
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text

# Determine stop words as the words that convery little to no information about the actual content like the words: the, of, for etc.
def remove_stopwords(word_tokens):
    filtered_sentence = [] 
    stop_words = stopwords.words('english')
    specific_words_list = ['char', 'u', 'hindustan', 'doj', 'washington'] 
    stop_words.extend(specific_words_list )
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return filtered_sentence


# Create function for lemmatization.
def lemmatize(x):
    lemmatizer = WordNetLemmatizer()
    return' '.join([lemmatizer.lemmatize(word) for word in x])
  
    
# Split a string, text into a list of tokens.
tokenizer = RegexpTokenizer(r'\w+')
def tokenize(x): 
    return tokenizer.tokenize(x)
  

import nltk
nltk.download('omw-1.4')


# Apply all of these functions to the our dataframe.
news_articles_df['combined_text'] = news_articles_df['combined_text'].map(clean_text)
news_articles_df['tokens'] = news_articles_df['combined_text'].map(tokenize)
news_articles_df['tokens'] = news_articles_df['tokens'].map(remove_stopwords)
news_articles_df['lems'] =news_articles_df['tokens'].map(lemmatize)


# Print the head to check the format and the working of the get_articles function.
news_articles_df.head()
print(len(news_articles_df))


# Find keywords using the rake algorithm from NLTK (rake is Rapid Automatic Keyword Extraction algorithm, and is used for domain independent keyword extraction)
news_articles_df['keywords'] = ""
for index,row in news_articles_df.iterrows():
    comb_text = row['combined_text']
    r = Rake()
    r.extract_keywords_from_text(comb_text)
    key_words_dict = r.get_word_degrees()
    row['keywords'] = list(key_words_dict.keys())
  

# Apply the fucntion to the dataframe.
news_articles_df['keywords'] = news_articles_df['keywords'].map(remove_stopwords)
news_articles_df['lems'] =news_articles_df['keywords'].map(lemmatize)

# Print the head to check the format and the working of the get_articles function.
news_articles_df.head()


# Save modified dataframe into csv file.
news_articles_df.to_csv('news_articles_clean.csv', index = False)


# Check the data once again to ensure no null value is present.
print(news_articles_df.isnull().sum())
news_articles_df.dropna(inplace=True)
print(news_articles_df.shape)


#  Create two columns in order to put calculated value there.
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()
results = []
news_articles_df['pol_score'] = ''
news_articles_df['label'] = 0

# Calculating the score for each new.
for i, row in news_articles_df.iterrows():
    pol_score = sia.polarity_scores(row['lems'])
    pol_score['headline'] = row['lems']
    results.append(pol_score)
    news_articles_df['pol_score'][i] = pol_score['compound']
    if pol_score['compound'] > 0.2:
        news_articles_df['label'][i] = 'positive'
    elif pol_score['compound'] < -0.2:
        news_articles_df['label'][i] = 'negative'
    else:
        news_articles_df['label'][i] = 'neutral'
    

# Check the data once again to ensure with the dataframe format
news_articles_df


# Save received datafram as csv file.
writer = pd.ExcelWriter('data_to_plot_dashboard.xlsx')
news_articles_df.to_excel(writer)
writer.save()
