import pandas as pd
import numpy as np
import re
import math
from sklearn.model_selection import train_test_split

word_probabilities = []


def email_subject_to_words(subject):                        # to be able to calculate the probabilities of words in spam
    subject = subject.lower()                               # and non-spam (ham) cases
    words = re.findall("[a-z0-9]+", subject)

    return set(words)


def calculate_probabilities(x_train, y_train):              # analogous to training function, we are setting the
    num_of_spam_messages = y_train.value_counts()[1]        # probabilities as weights in naive bayes
    num_of_ham_messages = len(x_train) - num_of_spam_messages

    word_count_dictionary = {}                              # calculating the number of times a word comes in spam cases
    for index in range(len(x_train)):                       # and in ham cases
        words_set = email_subject_to_words(x_train.iloc[index])
        for word in words_set:
            if word not in word_count_dictionary:
                word_count_dictionary[word] = [0, 0]

            if y_train.iloc[index] == 0:
                word_count_dictionary[word][1] += 1
            else:
                word_count_dictionary[word][0] += 1

    word_probabilities_list = []                            # once number of times is calculated now we convert it to
    for word in word_count_dictionary:                      # the probabilities which will act as likelihood of x
        p_of_x_given_y_ham = (word_count_dictionary[word][1] + 0.5) / (num_of_ham_messages + 1)
        p_of_x_given_y_spam = (word_count_dictionary[word][0] + 0.5) / (num_of_spam_messages + 1)
        word_probabilities_list.append([word, p_of_x_given_y_spam, p_of_x_given_y_ham])

    return word_probabilities_list


def check_mail_subject(subject, word_probabilities_local):
    subject_word_set = email_subject_to_words(subject)
    spam_probability = 0
    ham_probability = 0
    for word, p_of_x_given_y_spam, p_of_x_given_y_ham in word_probabilities_local:
        if word in subject_word_set:
            spam_probability += (np.log(p_of_x_given_y_spam))
            ham_probability += (np.log(p_of_x_given_y_ham))

    spam_probability = np.exp(spam_probability)
    ham_probability = np.exp(ham_probability)
    if spam_probability + ham_probability == 0:
        return 0
    else:
        return spam_probability / (spam_probability + ham_probability)


data_df = pd.read_csv('data/emailSubjects.csv')
x = data_df['text']
y = data_df['spam']

x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=0.1, random_state=14)
word_probabilities = calculate_probabilities(x_training, y_training)


def test_function(x_test, y_test, word_prob_list):
    correct_predicted = 0
    spam_predicted_ham = 0
    ham_predicted_spam = 0

    for i in range(len(x_test)):
        predicted_probability = check_mail_subject(x_test.iloc[i], word_prob_list)
        if predicted_probability > 0.5:
            if y_test.iloc[i] == 1:
                correct_predicted += 1
            else:
                ham_predicted_spam += 1
        else:
            if y_test.iloc[i] == 0:
                correct_predicted += 1
            else:
                spam_predicted_ham += 1
                # print(i, predicted_probability)

    return correct_predicted, ham_predicted_spam, spam_predicted_ham


correct_predicted_, ham_predicted_spam_, spam_predicted_ham_ = test_function(x_testing, y_testing, word_probabilities)
accuracy = (correct_predicted_ / len(y_testing)) * 100
print(f"Accuracy: {accuracy} %")




