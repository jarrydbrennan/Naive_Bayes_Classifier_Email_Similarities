from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

##EXPLORE
emails = fetch_20newsgroups()
#print(emails.target_names)

#focus on baseball and hockey. can it tell the difference?
emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'])
#print(emails.target_names)

#content of email at index 5
#print(emails.data[5])
#target of email at index 5 (0=baseball, 1=hockey)
#print(emails.target[5])

##TRAIN/TEST SETS
train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset='train', shuffle = True, random_state=168)
test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset='test', shuffle = True, random_state=168)

##COUNTING WORDS
counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

##NAIVE BAYES CLASSIFIER
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts,test_emails.target))

