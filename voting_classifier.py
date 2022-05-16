import pandas as pd
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from functions import Custom_Transformer,get_polar_words
from sklearn.naive_bayes import MultinomialNB
import sklearn.feature_extraction.text as sk_text

path = '/Users/nikhil/data/ML_examples/erdos/'

election_df = pd.read_csv(path+'us_election_2020_1st_presidential_debate.csv')


Tfidf = sk_text.TfidfVectorizer(lowercase=True, analyzer='word',stop_words='english',ngram_range=(1,1))
Transformer = Custom_Transformer('Vice President Joe Biden','President Donald J. Trump',1,0)
df_new = Transformer.fit_transform(election_df)
X = df_new['text']
y = df_new['target']
Voting_classifier = Pipeline([("Tfidf vectorizer",Tfidf),("Multinomial NB",MultinomialNB())])
Voting_classifier.fit(X,y)

wc1,wc2 = get_polar_words(Voting_classifier)

plt.subplot(1,2,1)
plt.imshow(wc1)
plt.xticks([])
plt.yticks([])
plt.xlabel('Won the election',fontsize=20)

plt.subplot(1,2,2)
plt.imshow(wc2)
plt.xticks([])
plt.yticks([])
plt.xlabel('Lost the election',fontsize=20)




