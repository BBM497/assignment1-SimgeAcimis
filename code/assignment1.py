# Created by Simge ACIMIŞ
# Created in March 10, 2019 / 02:57 am
# Natural Language Processing Course / Assignment 1
# AutHorship Detection
import re
import math


# read files from list
# removes punctuations from text and makes all letters lowercase
def read_clean_file(path_list):

    dictionary = {'hamilton': [], 'madison':[], 'unknown':[]}
    path_beginswith = "../Data/Assignment1/"

    for file in path_list:

        file = open(path_beginswith + str(file) + ".txt", "r")
        lines = []

        for line in file:

            line = re.sub(r'[^\w\s]', '', line).lower()
            line = line.rstrip('\n').split(' ')
            lines.append(line)

        if lines[0][0] == 'hamilton':
            dictionary['hamilton'].append(lines[1])
        if lines[0][0] == 'madison':
            dictionary['madison'].append(lines[1])
        if lines[0][0] == 'unknown':
            dictionary['unknown'].append(lines[1])

    return dictionary


# task 1 unigram model
def build_unigram(author, name):
    dict = {}
    print("building unigram...")
    for arr in author[name]:
        for word in arr:
            if word in dict:
                dict[word] += 1
            else:
                dict[word] = 1
    return dict


# task 1 bigram model
def build_bigram(author, name):
    # len - 1 for dönecek
    dict = {}
    print("building bigram...")
    for arr in author[name]:
        for i in range(len(arr) - 1):
            if arr[i] + ' ' + arr[i+1] in dict:
                dict[arr[i] + ' ' + arr[i+1]] += 1
            else:
                dict[arr[i] + ' ' + arr[i+1]] = 1
    return dict


# task 1 trigram model
def build_trigram(author, name):
    print('building trigram...')
    dict = {}
    for arr in author[name]:
        for i in range(len(arr) - 2):
            if arr[i] + ' ' + arr[i+1] + ' ' + arr[i+2] in dict:
                dict[arr[i] + ' ' + arr[i+1] + ' ' + arr[i+2]] += 1
            else:
                dict[arr[i] + ' ' + arr[i+1] + ' ' + arr[i+2]] = 1
    return dict


def read_sentences_from_file(file_path):
    with open(file_path, "r") as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]


def calculate_bigram_perplexity(text, d1, d2):
    print('calculating bigram perplexity...')
    total_log_of_probabilities_d1 = 0
    total_log_of_probabilities_d2 = 0
    dict_text = {}
    for word in text:
        dict_text[word] = 0

    # madison total sum of log
    for i in range(len(text) - 1):
        if text[i] + ' ' + text[i+1] in d1:
            total_log_of_probabilities_d1 += math.log2(d1[text[i] + ' ' + text[i+1]])
        else:
            total_log_of_probabilities_d1 += math.log2(1 / (len(d1) + len(text)))

    # hamilton total sum of log
    for i in range(len(text) - 1):
        if text[i] + ' ' + text[i+1] in d2:
            total_log_of_probabilities_d2 += math.log2(d2[text[i] + ' ' + text[i+1]])
        else:
            total_log_of_probabilities_d2 += math.log2(1 / (len(d2) + len(text)))


    return [1 / math.pow(2, (total_log_of_probabilities_d1 / len(text))),
            1 / math.pow(2, (total_log_of_probabilities_d2 / len(text)))]


def calculate_trigram_perplexity(text, d1, d2):
    print('calculating trigram perplexity...')
    total_log_of_probabilities_d1 = 0
    total_log_of_probabilities_d2 = 0
    dict_text = {}
    for word in text:
        dict_text[word] = 0
    # madison total sum of log
    for i in range(len(text) - 2):
        if text[i] + ' ' + text[i+1] + ' ' + text[i+2] in d1:
            total_log_of_probabilities_d1 += math.log2(d1[text[i] + ' ' + text[i+1] + ' ' + text[i+2]])
        else:
            total_log_of_probabilities_d1 += math.log2(1 / len(d1))  #laplace smoothing

    # hamilton total sum of log
    for i in range(len(text) - 2):
        if text[i] + ' ' + text[i+1] + ' ' + text[i+2] in d2:
            total_log_of_probabilities_d2 += math.log2(d2[text[i] + ' ' + text[i+1] + ' ' + text[i+2]])
        else:
            total_log_of_probabilities_d2 += math.log2(1 / len(d2)) # laplace smoothing

    return [1 / math.pow(2, total_log_of_probabilities_d1 / len(text)),
            1 / math.pow(2, total_log_of_probabilities_d2 / len(text))]

if __name__== "__main__":

    # task 1 -> build unigram, bigram, trigram language models
    # hamilton -> [1, 6, 7, 8, 13, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    # madison -> [10, 14, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

    # read and clean data from punctuations and make all letters lowercase
    task1_path = [1, 6, 7, 8, 10, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    task1_dict = read_clean_file(task1_path)


    madison = 'madison'
    hamilton = 'hamilton'


    # unigram, bigram, trigram language models for madison
    word_count_unigram_madison = build_unigram(task1_dict, madison)
    word_count_bigram_madison = build_bigram(task1_dict, madison)
    word_count_trigram_madison = build_trigram(task1_dict, madison)


    # unigram, bigram, trigram language models for hamilton
    word_count_unigram_hamilton = build_unigram(task1_dict, hamilton)
    word_count_bigram_hamilton = build_bigram(task1_dict, hamilton)
    word_count_trigram_hamilton = build_trigram(task1_dict, hamilton)


    # calculate word probabilities and update the values and add-one Laplace Smoothing
    dictionaries = [word_count_unigram_madison, word_count_bigram_madison, word_count_trigram_madison, word_count_unigram_hamilton, word_count_bigram_hamilton, word_count_trigram_hamilton]
    for dict in dictionaries:
        for element in dict:
            dict[element] = dict[element]  / len(dict)


    #  ------------------------- task 3 -------------------------------

    # labeled test data to decide which model is more accurate on predicting authors
    task3_path = [9, 11, 12, 47, 48, 58]
    task3_dict = read_clean_file(task3_path)


    # prediction step for labeled test data using bigram perplexity
    for text in task3_dict[madison]:
        # resultingArr -> tuple(madison_perplexity_value, hamilton_perplexity_value)
        resultingArr = calculate_bigram_perplexity(text, word_count_bigram_madison, word_count_bigram_hamilton)
        if resultingArr[0] < resultingArr[1]:
            print('Successful prediction for Madison Essay')
        else:
            print('Unsuccessful prediction for Madison Essay')
        print('----------------------------------------------')

    for text in task3_dict[hamilton]:
        resultingArr = calculate_bigram_perplexity(text, word_count_bigram_madison, word_count_bigram_hamilton)
        if resultingArr[0] > resultingArr[1]:
            print('Successful prediction for Hamilton Essay')
        else:
            print('Unsuccessful prediction for Hamilton Essay')
        print('----------------------------------------------')


    print('\n\n-------------------Bigram finished, Trigram Started-----------------------------\n\n')

    # prediction step for labeled test data using trigram perplexity
    for text in task3_dict[madison]:
        resultingArr = calculate_trigram_perplexity(text, word_count_trigram_madison, word_count_trigram_hamilton)
        if resultingArr[0] < resultingArr[1]:
            print('Successful prediction for Madison Essay')
        else:
            print('Unsuccessful prediction for Madison Essay')
        print('----------------------------------------------')

    for text in task3_dict[hamilton]:
        resultingArr = calculate_trigram_perplexity(text, word_count_trigram_madison, word_count_trigram_hamilton)
        if resultingArr[0] > resultingArr[1]:
            print('Successful prediction for Hamilton Essay')
        else:
            print('Unsuccessful prediction for Hamilton Essay')
        print('----------------------------------------------')

    # so I decided to use bigram perplexity, because it predicted 5 correct out of 6 essay
    # and When I used trigram perplexity, it predicted 3 correct out of 6 essay

    # task 3 unlabeled test data -> 49, 50, 51, 52, 53, 54, 55, 56, 57, 62, 63
    task3_unlabeled_test_path = [49, 50, 51, 52, 53, 54, 55, 56, 57, 62, 63]
    unlabeled_test = read_clean_file(task3_unlabeled_test_path)

    # prediction step for labeled test data using bigram perplexity
    unknown = 'unknown'
    for text in unlabeled_test[unknown]:
        # resultingArr -> tuple(madison_perplexity_value, hamilton_perplexity_value)
        resultingArr = calculate_bigram_perplexity(text, word_count_bigram_madison, word_count_bigram_hamilton)
        print(resultingArr)
        if resultingArr[0] < resultingArr[1]:
            print('Prediction -> Madison')
        else:
            print('Prediction -> Hamilton')
        print('----------------------------------------------')


"""
    print('unlabeled data trigram ------------------')
    # prediction step for labeled test data using trigram perplexity
    for text in unlabeled_test[unknown]:
        resultingArr = calculate_trigram_perplexity(text, word_count_trigram_madison, word_count_trigram_hamilton)
        print(resultingArr)
        if resultingArr[0] < resultingArr[1]:
            print('Prediction -> Madison')
        else:
            print('Prediction -> Hamilton')
       print('----------------------------------------------')
"""