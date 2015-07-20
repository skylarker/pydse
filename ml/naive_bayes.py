from collections import defaultdict, Counter
from nltk import *
import glob
import math
import re

import numpy as np


np.random.seed(123)


class NaiveBayesClassifier(object):
    def __init__(self, k_=0.5):
        self.data = []
        self.k = k_
        self.word_probs = []
        self.train_data = []
        self.test_data = []
        self.word_probs = []
        self.stemmer = LancasterStemmer()

    def tokenize(self, message):
        message = message.lower()
        #all_words = re.findall("[a-z0-9']+", message)
        all_words = nltk.word_tokenize(message)
        all_words = [self.stemmer.stem(word_) for word_ in all_words]
        return set(all_words)

    def traverse_files(self, path_):
        for fn_ in glob.glob(path_):
            is_spam_ = 'ham' not in fn_
            with open(fn_, 'r') as file_:
                for line in file_:
                    if line.startswith("Subject:"):
                        # remove the leading "Subject: " and keep what's left
                        subject_ = re.sub(r"^Subject: ", "", line).strip()
                        self.data.append((subject_, is_spam_))

    def count_words(self):
        """
            training set consists of pairs (message, is_spam)
        """
        counts_ = defaultdict(lambda: [0, 0])

        for message, is_spam_ in self.train_data:
            for word_ in self.tokenize(message):
                counts_[word_][0 if is_spam_ else 1] += 1
        return counts_

    def word_probabilities(self, counts_, total_spams, total_non_spams, k_=0.5):
        """
            turn the word_counts into a list of triplets
            w, p(w | spam) and p(w | ~spam)
        """
        for w, (spam, non_spam) in counts_.iteritems():
            self.word_probs.append((w, (spam + k_) / (total_spams + 2 * k_),
                                    (non_spam + k_) / (total_non_spams + 2 * k_)))
        return self.word_probs

    def spam_probability(self, word_probs, message):
        message_words = self.tokenize(message)
        log_prob_if_spam = log_prob_if_not_spam = 0.0
        # iterate through each word in our vocabulary
        for word_, prob_if_spam, prob_if_not_spam in word_probs:
            # if *word* appears in the message,
            # add the log probability of seeing it
            if word_ in message_words:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_not_spam += math.log(prob_if_not_spam)
            else:
                # if *word* doesn't appear in the message
                # add the log probability of _not_ seeing it # which is log(1 - probability of seeing it)
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_not_spam = math.exp(log_prob_if_not_spam)
        return prob_if_spam / (prob_if_spam + prob_if_not_spam)

    def classify(self, message):
        return self.spam_probability(self.word_probs, message)

    def split_data(self, cut_off=0.75):
        total = len(self.data)
        split_pivot = int(cut_off * total)
        np.random.shuffle(self.data)  # shuffle the data (simplest form of cross validation)
        self.train_data = self.data[:split_pivot]
        self.test_data = self.data[split_pivot:]

    def train(self):
        # count spam and non-spam messages
        num_spams = len([is_spam for message, is_spam in self.train_data if is_spam])
        num_non_spams = len(self.train_data) - num_spams
        # run training data through our "pipeline"
        word_counts = self.count_words()
        self.word_probabilities(word_counts, num_spams, num_non_spams, self.k)

    @staticmethod
    def p_spam_given_word(word_prob):
        """
            Uses Bayes' theorem to compute p(spam | message contains word)
        """
        # word_prob is one of the triplets produced by word_probabilities

        _, prob_if_spam, prob_if_not_spam = word_prob
        return prob_if_spam / (prob_if_spam + prob_if_not_spam)


if __name__ == '__main__':
    classifier = NaiveBayesClassifier()
    path = '../data/nb/*/*'  # wild carded path
    classifier.traverse_files(path)
    classifier.split_data()
    classifier.train()
    # triplets (subject, actual is_spam, predicted spam probability)
    classified = [(subject, is_spam, classifier.classify(subject)) for subject, is_spam in classifier.test_data]
    # assume that spam_probability > 0.5 corresponds to spam prediction # and count the combinations
    # of (actual is_spam, predicted is_spam)
    print '********* Results *********'
    counts = Counter((is_spam, spam_probability > 0.5) for _, is_spam, spam_probability in classified)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for is_spam_prediction, count in counts.iteritems():
        if is_spam_prediction[0] is True and is_spam_prediction[0] == is_spam_prediction[1]:
            print 'True Positive: ', count
            tp = count
        if is_spam_prediction[0] is False and is_spam_prediction[0] == is_spam_prediction[1]:
            print 'True Negative: ', count
            tn = count
        if is_spam_prediction[0] is True and is_spam_prediction[0] != is_spam_prediction[1]:
            print 'False Positive', count
            fp = count
        if is_spam_prediction[0] is False and is_spam_prediction[0] != is_spam_prediction[1]:
            print 'False Negative', count
            fn = count
    print
    print 'Precision = ', float(tp)/tp+fp
    print 'Recall = ', float(tp/tp+fn)
    print

    words = sorted(classifier.word_probs, key=classifier.p_spam_given_word)
    spammiest_words = words[-5:]
    print '******* Spammiest Words *******'
    for word, _, _ in spammiest_words:
        print word
    hammiest_words = words[:5]
    print
    print '******* Hammiest Words *******'
    for word, _, _ in hammiest_words:
        print word