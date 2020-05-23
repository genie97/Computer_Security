import os
import numpy as np
from collections import Counter

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC


def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:  # Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)

    list_to_remove = list(filter(lambda x: not x.isalpha() or len(x) == 1, dictionary.keys()))

    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    dictionary = dictionary.most_common(3000)
    return dictionary

def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix



# Create a dictionary of words with its frequency

train_dir = 'DATA/ling_spam/train-mails'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training SVM and Naive bayes classifier

model1 = MultinomialNB()
model2 = SVC()
model3 = GaussianNB()
model4 = BernoulliNB()
model5 = NuSVC()
model6 = LinearSVC()

model1.fit(train_matrix, train_labels)
model2.fit(train_matrix, train_labels)
model3.fit(train_matrix, train_labels)
model4.fit(train_matrix, train_labels)
model5.fit(train_matrix, train_labels)
model6.fit(train_matrix, train_labels)


# Test the unseen mails for Spam
test_dir = 'DATA/ling_spam/test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1

result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
result3 = model2.predict(test_matrix)
result4 = model2.predict(test_matrix)
result5 = model2.predict(test_matrix)
result6 = model2.predict(test_matrix)

print(confusion_matrix(test_labels, result1))
print(confusion_matrix(test_labels, result2))
print(confusion_matrix(test_labels, result3))
print(confusion_matrix(test_labels, result4))
print(confusion_matrix(test_labels, result5))
print(confusion_matrix(test_labels, result6))

# 모델1: 멀티노밍러 나이브 베이지안 모델
print("model1(Multinomial Naive Bayesian) accuracy_score: {}".format(accuracy_score(test_labels, result1)))
print("model1(Multinomial Naive Bayesian) precision_score: {}".format(precision_score(test_labels, result1)))

# 모델2: 서포트 벡터 머신 모델
print("model2(SVM) accuracy_score: {}".format(accuracy_score(test_labels, result2)))
print("model2(SVM) precision_score: {}".format(precision_score(test_labels, result2)))

# 모델3: 가우시안 나이브 베이지안 모델
print("model3(Gaussian Naive Bayesian) accuracy_score: {}".format(accuracy_score(test_labels, result2)))
print("model3(Gaussian Naive Bayesian) precision_score: {}".format(precision_score(test_labels, result2)))

# 모델4: 베르누이 나이브 베이지안 모델
print("model4(Bernoulli Naive Bayesian) accuracy_score: {}".format(accuracy_score(test_labels, result2)))
print("model4(Bernoulli Naive Bayesian) precision_score: {}".format(precision_score(test_labels, result2)))

# 모델5: Nu 서포트 벡터 머신 모델
print("model5(Nu SVC) accuracy_score: {}".format(accuracy_score(test_labels, result2)))
print("model5(Nu SVC) precision_score: {}".format(precision_score(test_labels, result2)))

# 모델6: Linear 서포트 벡터 머신 모델
print("model6(Linear SVC) accuracy_score: {}".format(accuracy_score(test_labels, result2)))
print("model6(Linear SVC) precision_score: {}".format(precision_score(test_labels, result2)))