import numpy as np
import lda
import pandas as pd
# Reads in list of words, converts to tuple
dfw = pd.read_csv('wordsA.csv')
dfw = dfw.transpose()
words = tuple(dfw.itertuples(index=True))
words = words[0]
# Reads in word distribution by HAR (encounter) matrix as numpy array
dfm = pd.read_csv('mainA.csv')
dfm = dfm.fillna(0)
dfm = dfm.astype('int64')
main = dfm.values
# Reads in HAR (encounter) details
dfe = pd.read_csv('encA.csv')
dfe = dfe.transpose()
enc = tuple(dfe.itertuples(index=True))
enc = enc[0]
# Choose specifics of model - # of topics, iterations, & words per topic; then it fits the model
model = lda.LDA(n_topics=29, n_iter=500, random_state=1)
model.fit(main)
topic_word = model.topic_word_
n_top_words = 4
# Creates the topics, outputs to console & csv file
topdata = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(words)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topdata.append([i, ', '.join(topic_words)])
    print('Topic {}: {}'.format(i, ', '.join(topic_words)))
topout = pd.DataFrame(topdata)
topout = topout.rename(columns={0: 'Topic', 1: 'Top Words'})
topout.to_csv('Topics_Detail.csv', index=False)
# Assigns the topics, outputs to console & csv file
data = []
doc_topic = model.doc_topic_
for i in range(len(enc)):
    data.append([enc[i],doc_topic[i].argmax(),max(doc_topic[i])])
    print("{} (Top Topic: {}, % Sure: {})".format(enc[i], doc_topic[i].argmax(), round((max(doc_topic[i])*100),2)))
dfout =  pd.DataFrame(data)
dfout = dfout.rename(columns={0: 'HAR', 1: 'Topic', 2: "% Likelihood"})
dfout.to_csv('HAR_Topics.csv', index=False)