import numpy as np
import lda
import pandas as pd
import matplotlib.pyplot as plt
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
arr=[]
# sets lower, upper bounds for topic counts
r1, r2 = 10,61
for n in (range(r1,r2)):
    model = lda.LDA(n_topics=n, n_iter=30, random_state=n)
    model.fit(main)  # model.fit_transform(main) is also available
    topic_word = model.topic_word_  # model.components_ also works
    arr.append(
        (min([max(topic_word[i]) for i in range(model.n_topics)]),
        max([max(topic_word[i]) for i in range(model.n_topics)]))
    )
    print arr

plt.plot(arr)
plt.title('Min,Max of Max Topic Word Prob')
plt.show()
