from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import math
import string
string.punctuation
import inflect
p = inflect.engine()
import nltk
nltk.download()

dataset = []
all_txt_files = []
for file in Path("txt").rglob("*.txt"):
     all_txt_files.append(file.parent / file.name)

for txt_file in all_txt_files:
    with open((txt_file),encoding="utf-8",errors='ignore') as f:
        txt_file_as_string = f.read()
    dataset.append(txt_file_as_string)

N = len(dataset)

# defining the function to convert lower case
def convert_lower_case(data):
    new_data = str(np.char.lower(str(data)))
    return new_data

# remove stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = str([word for word in word_tokens if word not in stop_words])
    return filtered_text

# defining the function to remove punctuation
def remove_punctuation(data):
    punctuationfree = "".join([i for i in data if i not in string.punctuation])
    return punctuationfree

# defining the function to remove numbers
def remove_number(data):
    result = ''.join([i for i in data if not i.isdigit()])
    return result

# function for lemmatization
def lemmatize_word(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmas = str([lemmatizer.lemmatize(word, pos ='v') for word in word_tokens])
    return lemmas

# function to preprocess the raw documents
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_number(data)
    data = remove_punctuation(data)
    data = remove_stopwords(data)
    data = remove_punctuation(data)
    data = lemmatize_word(data)
    data = remove_punctuation(data)
    return data

# function to calculate term frequency
def termFreq(word_dict, words):
    tf = {}
    numofWords = len(words)
    for word, count in word_dict.items():
        tf[word] = count/float(numofWords)
    return tf

# function to calculate TF-IDF scores
def TFIDFscore(tf, idf):
    tfidf = {}
    for word, val in idf.items():
        if word in tf.keys():
            tfidf[word] = val * tf[word]
        else:
            tfidf[word] = 0
    return tfidf

# function to calculate cosine similarity
def cosine_similarity(v1, v2):
    intr = set(v1.keys()) & set(v2.keys())
    num = sum([v1[x] * v2[x] for x in intr])
    sumv1 = sum([v1[x] ** 2 for x in list(v1.keys())])
    sumv2 = sum([v2[x] ** 2 for x in list(v2.keys())])
    denom = math.sqrt(sumv1) * math.sqrt(sumv2)
    if not denom:
        return 0.0
    else:
        return float(num)/denom

# function to vectorize text
def vectorize(text):
    return Counter(text)

processed_text = []
df = defaultdict(int)
idf = dict()
tf_idf = dict()

for txt_file in all_txt_files:
    with open((txt_file),encoding="utf-8",errors='ignore') as f:
        for line in f.readlines():
            # preprocessing step of all documents line by line
            processed_text.append(word_tokenize(str(preprocess(line))))

for j in range(N):
    tokens = processed_text[j]

    # calculation of document frequencies for each term
    for word in (set(tokens)):
        df[word] += 1

    # sorting dictionary according to document frequencies
    doc_freq = pd.DataFrame(df.items(), columns=['Word', 'Document Frequency']).sort_values(by='Document Frequency',ascending=False).reset_index(drop=True)

print("Document Frequency: ")
print(doc_freq)

# saving document frequencies as csv file to output file directory
Path("./DF_output").mkdir(parents=True, exist_ok=True)
DFoutput = [str(txt_file).replace(".txt", ".csv").replace("txt", "DF_output").replace("Document9","DF") for txt_file in all_txt_files]
doc_freq.to_csv(DFoutput[j])


for i in range(N):
    tokens = processed_text[i]
    dictionary = dict.fromkeys(tokens, 0)

    # calculation of term frequencies for each term
    for word in tokens:
        dictionary[word] += 1
        tf = termFreq(dictionary, tokens)

    # sorting dictionary according to term frequencies
    term_freq = pd.DataFrame(tf.items(), columns=['Word', 'Term Frequency']).sort_values(by='Term Frequency', ascending=False).reset_index(drop=True)

    # saving term frequencies as csv file to output file directory
    Path("./TF_output").mkdir(parents=True, exist_ok=True)
    TFoutput = [str(txt_file).replace(".txt", ".csv").replace("txt", "TF_output") for txt_file in all_txt_files]
    term_freq.to_csv(TFoutput[i])

    print("TF of document ID: ",i)
    print(term_freq)

    # calculation of inverse document frequencies for each term
    for word in df:
        idf[word] = math.log10(N / float(df[word]))

    # sorting dictionary according to inverse document frequencies
    id_freq = pd.DataFrame(idf.items(), columns=['Word', 'Inverse Document Frequency']).sort_values(by='Inverse Document Frequency', ascending=False).reset_index(drop=True)

    # calculation tf-idf scores for each term
    for word in tokens:
        tf_idf = TFIDFscore(tf,idf)

    # sorting dictionary according to tf-idf scores
    tfidf = pd.DataFrame(tf_idf.items(), columns=['Word', 'TF-IDF']).sort_values(by='TF-IDF', ascending=False).reset_index(drop=True)

    print("TFIDF of document ID: ", j)
    print(tfidf)

    # saving TFIDFs as csv file to output file directory
    Path("./TFIDF_output").mkdir(parents=True, exist_ok=True)
    TFIDFoutput = [str(txt_file).replace(".txt", ".csv").replace("txt", "TFIDF_output") for txt_file in all_txt_files]
    tfidf.to_csv(TFIDFoutput[i])

# saving inverse document frequencies as csv file to output file directory
Path("./IDF_output").mkdir(parents=True, exist_ok=True)
IDFoutput = [str(txt_file).replace(".txt", ".csv").replace("txt", "IDF_output").replace("Document9","IDF") for txt_file in all_txt_files]
id_freq.to_csv(IDFoutput[i])


# calculation of cosine similarities between documents
for a in range(N):
    v1 = vectorize(processed_text[a])
    for b in range(a + 1, N):
        v2 = vectorize(processed_text[b])
        cos = cosine_similarity(v1, v2)
        print("The cosine similarity between the document ID: ", a, "and", b, "is: ",cos)

