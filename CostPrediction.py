from sklearn.model_selection import train_test_split
import pandas as pd
#Classifier imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Performance metrics imports
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv('DIAGmainA.csv')
dfclass = pd.read_csv('encA.csv')
#Check that data is loaded correctly
print df.head(3)
print type(df)
print df.shape[0]
print dfclass.head(3)
print type(dfclass)
print dfclass.shape[0]

# separate feature labels for classifiers & fill NaNs in features.
features = df.iloc[:,0:9097]
labels = dfclass['CostClass']
features = features.apply(pd.to_numeric)
features = features.fillna(0)
features[features != 0] = 1

#scaler = MinMaxScaler()
#features = scaler.fit_transform(features)
#test scaling with above lines

# Training/ Test Split
x1,x2,y1,y2 =train_test_split(features, labels, random_state=0, train_size =0.2)

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)

# Initialize our classifiers

GNB = GaussianNB()
KNN = KNeighborsClassifier(n_neighbors=5)
BNB = BernoulliNB()
LR = LogisticRegression()
SGD = SGDClassifier(max_iter=5, tol=None)
LSVC = LinearSVC()

# Train our classifier and test predict
GNB.fit(x1, y1)
y2_GNB_model = GNB.predict(x2)
print "GaussianNB Accuracy :", round((accuracy_score(y2, y2_GNB_model)*100),2), "%"

KNN.fit(x1,y1)
y2_KNN_model = KNN.predict(x2)
print "KNN Accuracy :", round((accuracy_score(y2, y2_KNN_model)*100),2), "%"

BNB.fit(x1,y1)
y2_BNB_model = BNB.predict(x2)
print "BNB Accuracy :", round((accuracy_score(y2, y2_BNB_model)*100),2), "%"

LR.fit(x1,y1)
y2_LR_model = LR.predict(x2)
print "LR Accuracy :", round((accuracy_score(y2, y2_LR_model)*100),2), "%"

SGD.fit(x1,y1)
y2_SGD_model = SGD.predict(x2)
print "SGD Accuracy :", round((accuracy_score(y2, y2_SGD_model)*100),2), "%"

LSVC.fit(x1,y1)
y2_LSVC_model = LSVC.predict(x2)
print "LSVC Accuracy :", round((accuracy_score(y2, y2_LSVC_model)*100),2), "%"


################################# Display Confusion Matrix for most accurate algorithm #############################################
# Encounters are grouped into cost deciles
target_names = ['1','2','3','4','5','6','7','8','9','10']
print(classification_report(y2, y2_LR_model, target_names=target_names, sample_weight=None, digits=4))

import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
class_names = target_names
# This function prints and plots the confusion matrix
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y2, y2_LR_model)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()