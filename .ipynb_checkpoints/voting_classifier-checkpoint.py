import pandas as pd
import os
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
import sklearn.feature_extraction.text as sk_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
os.chdir('/Users/nikhil/code/Erdos/Project')
from functions import Custom_Transformer,get_polar_words

path = '/Users/nikhil/data/ML_examples/erdos/'

#combine 1st and second presidential debates for 2020
election_df_1 = pd.read_csv(path+'us_election_2020_1st_presidential_debate.csv')
election_df_2 = pd.read_csv(path+'us_election_2020_2nd_presidential_debate.csv')
election_df = pd.concat([election_df_1,election_df_2],axis=0)

#train test split
train_election,test_election = train_test_split(election_df.copy(),test_size=0.2, random_state=42)


#Set True for bigrams otherwise set False for unigrams
bigrams = False
if bigrams:
   ngrams = (2,2)
else:
   ngrams = (1,1)
   
#define tdif vectorizer
Tfidf = sk_text.TfidfVectorizer(lowercase=True,analyzer='word',stop_words= 'english',ngram_range=ngrams)
#define the transformer(this adds a column called target which is 1 if the candidate won and 0 if lost)
Transformer = Custom_Transformer('Vice President Joe Biden','President Donald J. Trump',1,0)
#transform train and tests data sets
df_new_train = Transformer.fit_transform(train_election)
df_new_test =  Transformer.fit_transform(test_election)

#separate X and y for train and test
X = df_new_train['text']
y = df_new_train['target']
X_test = df_new_train['text']
y_test = df_new_train['target']

#define multinomial NB
Voting_classifier = Pipeline([("Tfidf vectorizer",Tfidf),("Multinomial NB",MultinomialNB())])
#fit on training data
Voting_classifier.fit(X,y)
#predict on test data
y_predict = Voting_classifier.predict(X_test)


#generate word clouds top 50 polar words (negative=>lost election/positive=>won election)
wc1,wc2,feature_names = get_polar_words(Voting_classifier,50)


#plot polar words
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

#plot confusion matrix
plot_confusion_matrix(Voting_classifier, X_test, y_test)


