#This is a model that will analyze the emotion of the text.
#We will be using GoEmotion dataset: https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/
#https://github.com/google-research/google-research/tree/master/goemotions

#start off with importing pandas to read the dataset
import pandas as pd

#read the dataset
df1 = pd.read_csv("C:/Users/reyno/moodjournal/models/test_models/data/full_dataset/goemotions_1.csv")
df2 = pd.read_csv("C:/Users/reyno/moodjournal/models/test_models/data/full_dataset/goemotions_2.csv")
df3 = pd.read_csv("C:/Users/reyno/moodjournal/models/test_models/data/full_dataset/goemotions_3.csv")

#concatenate the datasets, this will give us a single dataframe
combineddf = pd.concat([df1, df2, df3])

#print the first 5 rows of the dataset
print(combineddf.head())

#print the last 5 rows of the dataset
print(combineddf.tail())

#print the shape of the dataset
print(combineddf.shape)
#check the columns of the dataset
print(combineddf.columns)

combineddf.drop(columns=["id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id"],inplace=True)

print(combineddf.columns)