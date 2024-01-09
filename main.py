import ndjson
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
with open('Video_Games_5.json') as f:
    data = ndjson.load(f)
reviews_df = pd.DataFrame(data)
print(reviews_df.head())
print(reviews_df.shape)
print(reviews_df.info())
sns.countplot(data = reviews_df, x='overall')
plt.show()
print(len(reviews_df['asin'].value_counts(dropna=False)))
one_1500 = reviews_df[reviews_df['overall']==1.0].sample(n=1500)
two_500 = reviews_df[reviews_df['overall']==2.0].sample(n=500)
three_500 = reviews_df[reviews_df['overall']==3.0].sample(n=500)
four_500 = reviews_df[reviews_df['overall']==4.0].sample(n=500)
five_1500 = reviews_df[reviews_df['overall']==5.0].sample(n=1500)
undersampled_reviews = pd.concat([one_1500, two_500, three_500, four_500, five_1500], axis=0)
print(undersampled_reviews['overall'].value_counts(dropna=False))
sns.countplot(data=undersampled_reviews, x='overall')
sns.countplot(data = reviews_df, x='overall')
plt.show()
sample_100K_revs = reviews_df.sample(n=100000, random_state=42)
undersampled_reviews.to_csv("small_corpus.csv", index=False)
sample_100K_revs.to_csv("big_corpus.csv", index=False)