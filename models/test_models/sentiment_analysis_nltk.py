#we will create a sentiment analysis model using nltk
#ultimatley this is read the journal entries and analyze the sentiment of the entry

#we will use the vader sentiment analyzer from nltk


import nltk
from nltk import data as nltk_data

def ensure_vader_lexicon():
    try:
        nltk_data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def vader_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    
    scores = analyzer.polarity_scores(text)

    return scores

def score_analysis(scores):
    
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"
    
    

if __name__ == "__main__":
    ensure_vader_lexicon()
    text = input("Enter a text to analyze: ")
    scores = vader_sentiment_analysis(text)
    print(scores)
    print(score_analysis(scores))
    
    