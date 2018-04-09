# First install the NLTK package 
import pip 
pip.main(['install', "nltk"])

# Then download the train data. This is necessay because the NLTK comes with many corpora, toy grammars, trained models, etc.
import nltk
nltk.download()
# A window will pop-up. Please, download the entire collection (using "all")

# N.B. This is a quick tutorial for the NLTK package, and the NLP (Natural language Processing) itself. 
# It's not a complete course on the NLTK, because it's a large subject, wich has to be studied not just via this tutorial, but rather with the following book : http://www.nltk.org/book
# This is more a tutorial for data scientist who is interested by the textmining. 

# From a data analyst/scientist point of view the best way to store one (or several) text corpuse(s) will be a list (array).

# First step in the textmining process is the tokenization
# The reason behind it is to copy the human way to undestand text -> we understand it word by word (sentence by sentence)
# In order to do that, NLTK propose the tokenize function (by word or by sentence)
from nltk.tokenize import sent_tokenize, word_tokenize

# Import dummy data
data = open(r"C:\Users\VictorVrabie\Speech_Adi.txt").read()

# Tokenize the text
#print(word_tokenize(data))
#print(sent_tokenize(data))


# When analyzing the first output list (the word-tokenized) we can see that there are a lot of words that did not give us much information 
# Those are the so called stopwords. We will delete them from the corpus. 

from nltk.corpus import stopwords
stopWords = stopwords.words('english')

# If we want to add specific stopwords, we wil just append them to the initial list of stopwords
newStopWords = ['communism','Lenin','Marx','Sviet Union','commarade','Commumist Manifesto', 'mr.']
stopWords_new = stopWords + newStopWords 

words = word_tokenize(data)
wordsFiltered = []

# Then we take the words one by one and compare them to the list of stop words
# For the sake of simplicity we will work from now with homogeneous lowercase text 
for w in words:
    if w.lower() not in stopWords:
        wordsFiltered.append(w.lower())

# We will join the text back, to see what will happen if the text doesn't contain the stopwords
data_new = ' '.join(map(str,wordsFiltered))
#print(data_new)

# The next classic step in the text analysis, will be the stemming(lemmisation)
# Even if there are several stemmers, we will first define rudimental stemmer
def stem(word):
     for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment','th']:
         if word.endswith(suffix):
             return word[:-len(suffix)]
     return word

words = word_tokenize(data_new)
# Remove words that are shorter than 2 characters 
words = [word for word in words if len(word) > 2]
# Remove numbers
words = [word for word in words if not word.isnumeric()]

wordsStemmed = []

for w in words:
    wordsStemmed.append(stem(w))
    
data_stemmed = ' '.join(map(str,wordsStemmed))
#print(data_stemmed)

# =============================================================================
# # Alternatively we can use predefined Stemmer
# from nltk.stem.snowball import SnowballStemmer
# stemmer = SnowballStemmer("english",ignore_stopwords=True)
# 
# words = word_tokenize(data_new)
# wordsStemmed = []
# 
# for w in words:
#     wordsStemmed.append(stemmer.stem(w))
#     
# data_stemmed = ' '.join(map(str,wordsStemmed))
# print(data_stemmed)
# =============================================================================

# Once that we have the clean text, we can work on the basic descriptive statistics
# Let's get words frequencies
from nltk import FreqDist, bigrams, trigrams
fdist = FreqDist(wordsStemmed)
# Print 10 most used words
print(fdist.most_common(10))

# or let's say the bigrams
fdist_bigrams = FreqDist(list(bigrams(wordsStemmed)))
# Print 10 most used words
print(fdist_bigrams.most_common(10))

# or even trigrams
fdist_trigrams = FreqDist(list(trigrams(wordsStemmed)))
# Print 10 most used words
print(fdist_trigrams.most_common(10))


# =============================================================================
# # Optionally we can create visualization on the text. Usually those are wordclouds.
# # Here you cand find an example how to do one, in the form of a specific image(silhouette)
# # Vizualization inspired from here http://ramiro.org/notebook/sherlock-holmes-canon-wordcloud/
# import random
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# 
# from nltk.corpus import stopwords
# from scipy.misc import imread
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
# from wordcloud import WordCloud, STOPWORDS
# 
# mpl.style.use('classic')
# limit = 1000
# infosize = 12
# 
# title = 'Most frequent words in the speech of Adolf Hitler'
# chartinfo = 'Author: Victor Vrabie'
# footer = 'The {} most frequent words used in the Adolf Hitler speech at 4th anniverssay of comming to power .\n{}'.format(limit, chartinfo)
# font = '/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf'
# fontcolor='#fafafa'
# bgcolor = '#000000'
# english_stopwords = set(stopwords.words('english')) | STOPWORDS | ENGLISH_STOP_WORDS
# 
# 
# def grey_color(word, font_size, position, orientation, random_state=None, **kwargs):
#     return 'hsl(0, 0%%, %d%%)' % random.randint(50, 100)
# 
# wordcloud = WordCloud(
#     max_words=limit,
#     stopwords=english_stopwords,
#     mask=imread(r'C:\Users\VictorVrabie\Downloads\silhouette.png'),
#     background_color=bgcolor
# ).generate(data_stemmed)
# 
# fig = plt.figure()
# fig.set_figwidth(21)
# fig.set_figheight(27)
# 
# plt.imshow(wordcloud.recolor(color_func=grey_color, random_state=3))
# plt.title(title, color=fontcolor, size=30, y=1.01)
# plt.annotate(footer, xy=(0, -.025), xycoords='axes fraction', fontsize=infosize, color=fontcolor)
# plt.axis('off')
# plt.show()
# 
# plt.savefig('plot.png')
# 
# =============================================================================

# There might interesting to analyse the structure 
# That means that we can analyse each word in the text and tag it by it's part-of-speech class (Conjunction, preposition, personal pronoun, verb etc.)  
import nltk

# We will analyse the structure of sentences
sentences = nltk.sent_tokenize(data)  
sentencePunkt = [] 

#Then the function of each word in that specific sentence
for sent in sentences:
    sentencePunkt.append(nltk.pos_tag(nltk.word_tokenize(sent)))

# Then just print the first sentence that was decontructed and tagged word by word
# print(sentencePunkt[0])

# In order to understand what thoe "NN", "IN", "DT" mean please take a look https://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/
# There are some better sources of it (the NLTK book that I posted before), but this one is the shortest and the best one.

# This classification will give us the opportunity to analyze the words as part of speech, thus "on va pas melanger des choux et de carrottes"
# This means that we get the frequences for the comparative adjective or proper nouns etc.

def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(10)) for tag in cfd.conditions())

sentencePunkt_bis = sum(sentencePunkt,[])

# Let's say we want to know 10 most used proper nouns
tagdict = findtags('NNP', sentencePunkt_bis)
for tag in sorted(tagdict):
     print(tag, tagdict[tag])
     
# or comparative adjective
tagdict = findtags('JJR', sentencePunkt_bis)
for tag in sorted(tagdict):
     print(tag, tagdict[tag])

# or verb
tagdict = findtags('VB', sentencePunkt_bis)
for tag in sorted(tagdict):
     print(tag, tagdict[tag])


# =============================================================================
# Next thing that will be interesting to analyse will be the TF-IDF index. As the name means it (Term frequency * Inverse document Frequency)
# This index will compute the index of "importance" or the "meaninless" of a word in a corpus and a selection of documents.
# EX: In a selection of documents about comunism, the word "communism" will apear everywhere, but it does not mean that it is "interesting" to analyze it.
# Instead we will analyse the words that are specific for every document and not the entire corpus.
# Please find more information about tf-idf https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# However there is no dedicated function for tf-idf in the nltk package, and we have to create the functions by ourself.
# I will provide however the way to do it but it's not the cleanest one, and by far, not the fastest one.


import math

data_josy = open(r"C:\Users\VictorVrabie\Speech_Josy.txt").read()

# Define the function we will use lates
def freq(word, doc):
    return doc.count(word)

def word_count(doc):
    return len(doc)

def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))

def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))

def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))


#Compute the frequency for each term.
vocabulary = []
docs = {}
all_tips = []

#take te list of your documents, tockenize and remove the stopwords
for tip in ([data, data_josy]):
    tokens = word_tokenize(tip)

    tokens = [token.lower() for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in stopwords]

    final_tokens = []
    final_tokens.extend(tokens)
    docs[tip] = {'freq': {}, 'tf': {}, 'idf': {},
                        'tf-idf': {}, 'tokens': []}
    
    
    for token in final_tokens:
        #The frequency computed for each tip
        docs[tip]['freq'][token] = freq(token, final_tokens)
        #The term-frequency (Normalized Frequency)
        docs[tip]['tf'][token] = tf(token, final_tokens)
        docs[tip]['tokens'] = final_tokens

    vocabulary.append(final_tokens)

# Compute the tf_idf
for doc in docs:
    for token in docs[doc]['tf']:
        #The Inverse-Document-Frequency
        docs[doc]['idf'][token] = idf(token, vocabulary)
        #The tf-idf
        docs[doc]['tf-idf'][token] = tf_idf(token, docs[doc]['tokens'], vocabulary)

#Find out the most relevant words by tf-idf.
words = {}
for doc in docs:
    for token in docs[doc]['tf-idf']:
        if token not in words:
            words[token] = docs[doc]['tf-idf'][token]
        else:
            if docs[doc]['tf-idf'][token] > words[token]:
                words[token] = docs[doc]['tf-idf'][token]

    print(doc)
    for token in docs[doc]['tf-idf']:
        print (token, docs[doc]['tf-idf'][token])

for item in sorted(words.items(), key=lambda x: x[1], reverse=False):
    print ("%f <= %s" % (item[1], item[0]))
# =============================================================================

# Using all this information we can provide a good descriptive analysis of a text.
# However, the next stpe will be to analyse the sentiments and to work on text classification.






































