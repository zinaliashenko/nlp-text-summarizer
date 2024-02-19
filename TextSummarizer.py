import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest # func allows to find the largest elements from an iterable object
from string import punctuation


'''This code is designed to generate an automatic text summary 
that highlights the most important sentences from the input text.'''

def summarize_text(text):

    # download lists of stopwords and puctuation
    
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    punctuations = punctuation + '\n'

    # tokenize the text by word
    token_words = nltk.word_tokenize(text)

    # find the frequencies for each word in the text
    # put them in the dict where key is a word, value is a frequency for the word
    word_frequencies = {}

    for word in token_words:
        if word.lower() not in stop_words:
            if word.lower() not in punctuations:
                if word.lower() not in word_frequencies.keys():
                    word_frequencies[word.lower()] = 1
                else:
                    word_frequencies[word.lower()] += 1

    # find the ration of word frequency to maximum word frequency (word_weight)
    # exchange word frequency with word_weight
    max_frequency = max(word_frequencies.values())

    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # tokenize text by sentence
    token_sentences = nltk.sent_tokenize(text)

    # find the score of each sentence in the text
    # put them into dict where key is a sentence, value is a sum of weights for every word in the sentence
    sentence_scores = {}

    for sentence in token_sentences:
        for word_weight in word_frequencies:
            if word_weight in sentence.lower():
                if sentence.lower() in sentence_scores:
                    sentence_scores[sentence] += word_frequencies[word_weight]
                else:
                    sentence_scores[sentence] = word_frequencies[word_weight]

    # length summary
    select_length = int(len(sentence_scores) * 0.3)

    # list of the largest elements from iterable object
    # nlargest(n, iterable, key=None)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    # join the list elements
    summary = ' '.join(summary)

    return summary


if __name__ == "__main__":
    
    file = 'data.txt'
    

    # load the text into variable
    with open(file, 'r', encoding='utf-8') as input_file:
        text = input_file.read()

    summary = summarize_text(text)

    # save the summary to a file
    with open('summary.txt', 'w') as output_file:
        output_file.write(summary)

