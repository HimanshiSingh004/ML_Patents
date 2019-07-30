from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import time


def get_labels(file):
    labels_name = []
    with open(file) as content_file:
        for line in content_file:
            words = line.split()
            labels_name.append(words[1])
    labels_name = list(set(labels_name))
    return labels_name


def read_data(file, data, labels_list, label_names):
    with open(file) as content_file:
        for line in content_file:
            words = line.split()
            text = ' '.join(word for word in words[2:] if len(word)>2 and len(word) <30)
            data.append(text)
            i = 0
            for label in label_names:
                if words[1]==label:
                    labels_list.append(i)
                i = i + 1


main_dir = 'F:/datafiles/patent_data/'
train_file = main_dir + 'patents_train.txt'
test_file = main_dir + 'patents_test.txt'

train_corpus = []
train_labels_list = []

train_labels_name = get_labels(train_file)
read_data(train_file, train_corpus, train_labels_list, train_labels_name)
train_labels = np.asarray(train_labels_list)

vec = TfidfVectorizer(stop_words='english', min_df=2, norm='l2')
train_corpus_fit = vec.fit_transform(train_corpus)

test_corpus = []
test_labels_list = []
test_labels_name = []
test_labels_name = get_labels(test_file)
read_data(test_file, test_corpus, test_labels_list, test_labels_name)
test_labels = np.asarray(test_labels_list)


test_corpus_fit = vec.transform(test_corpus)

start = time.time()
classifier_nb = MultinomialNB().fit(train_corpus_fit, train_labels)
predicted_nb = classifier_nb.predict(test_corpus_fit)
stop = time.time()

predicted_prob = classifier_nb.predict_proba(test_corpus_fit)
num_test_file = predicted_nb.size

output_file = 'output.txt'
Id = []

with open(test_file, 'r', encoding = 'UTF-8') as test:
    for line in test:
        Id.append(line.split()[0])

with open(output_file, 'w', encoding='UTF-8') as file:
    for num in range(0, num_test_file):
        sort_index = np.argsort(predicted_prob[num][:])[::-1][:3]
        data_tuple = str(Id[num]) + ' '
        for num2 in sort_index:
            predict_label = train_labels_name[num2]
            data_tuple += predict_label + ' '
        data_tuple += '\n'
        file.write(data_tuple)


print('\n Accuracy = ' + str(np.mean(predicted_nb == test_labels)))
print('\nconfusion matrix:')
print(metrics.confusion_matrix(test_labels, predicted_nb))
print('\nPerformance:')
print(metrics.classification_report(test_labels, predicted_nb, target_names=get_labels(test_file)))
print('\n\n')
print('Training + test_time = ' + str(stop-start))