# -*- coding: utf-8 -*-

import nltk
import sklearn
import string
import random
import pickle
import numpy

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')

def formatdata(formatted_sentences,formatted_labels,file_name):
    #file=open("en-ud-dev.conllu","r")
    file=open(file_name, 'r', encoding='ascii', errors='backslashreplace')
    #file=open(file_name,"rb")
    print("Reading data...")
    #quit()
    text=file.read().splitlines()
    tokens=[]
    labels=[]
    for line in text:
        line=line.split('\t')
        # print(line)
        if len(line)==3:
            tokens.append(line[0]) #Word itself
            if line[1]=="PUNCT": #Punctuation
                labels.append(line[0]+"P")
            else:
                labels.append(line[2])	#More detailed POS
        else:
            formatted_sentences.append(tokens)
            formatted_labels.append(labels)
            tokens=[]
            labels=[]





def creatdict(sentence,index,pos):	#pos=="" <-> featuresofword  else, relative pos (str) is pos
    word=sentence[index]
    wordlow=word.lower()
    word_root = stemmer.stem(wordlow)
    length = len(wordlow)
    sentence_len=len(sentence)
    suffix = wordlow[len(word_root):]
    prev_word = sentence[index - 1] if index != 0 else "<s>"
    next_word = sentence[index+1] if index != sentence_len-1  else "</s>"
    
    feature_set2={
        "wrd_low"+pos:wordlow,
        "cap"+pos:word[0].isupper(),
        "allcap"+pos:word.isupper(),
        "caps_inside"+pos:word==wordlow,
        "nums?"+pos:any(i.isdigit() for i in word),
        "word"+pos:word,                                                #word itself
        "root"+pos:word_root,                                           # word_root
        "suffixes"+pos:suffix,                                          # suffixes
        "len"+pos:length,                                               # length of word
        "stop_word"+pos:wordlow in english_stopwords,                   # is stop word
        "punc?"+pos:wordlow in string.punctuation,                      # is punctuation?
        "is_first"+pos:index==0,                                        # is first word in sentence
        "is_last"+pos:index==sentence_len-1,                           	# is last word in sentence
        "prev_word"+pos : prev_word,                                    # previous word
        "next_word"+pos: next_word,                                     # next word
    }
    return feature_set2


def feature_extractor(sentence,index):
    features=creatdict(sentence,index,"")
    return features




def creatsets(file_name):
    sentences=[]
    labels=[] 	#y_train (will be)
    formatdata(sentences,labels,file_name)
    limit=int(len(sentences))##############**********CHANGE these. these just limit the size of training set for faster trials. #####################
    sentences=sentences[:limit]##############
    labels=labels[:limit]####################

    #print(len(sentences),len(labels))
    # print(sentences)
    # print(labels)

    print("Feature extraction...")
    features=[]		#X_train
    for i in range(0,len(sentences)):
        features.append([])
        for j in range(0,len(sentences[i])):
            features[-1].append(feature_extractor(sentences[i],j))
            

    del sentences[:]
    del sentences


    delimit=int((len(labels)*8)/10)
    test_data=[features[delimit:],labels[delimit:]]
    features=features[:delimit]
    labels=labels[:delimit]

    training_data=[features,labels]


    with open('pos_crf_train.data', 'wb') as file:
        pickle.dump(training_data, file)
    file.close()


    with open('pos_crf_test.data', 'wb') as file:
        pickle.dump(test_data, file)
    file.close()

    return training_data, test_data



def train(training_data):
    print("Training...")
    features=training_data[0]
    labels=training_data[1]
    classifier.fit(features,labels)




def test(test_data):
    print("Testing...")

    y_true=test_data[1]  #labels
    y_pred=classifier.predict(test_data[0])

    #print(y_pred[0])

    precision=sklearn_crfsuite.metrics.flat_precision_score(y_true, y_pred,average='micro')
    recall=sklearn_crfsuite.metrics.flat_recall_score(y_true, y_pred,average='micro')
    f1=2*(precision*recall)/(precision+recall)
    accuracy=sklearn_crfsuite.metrics.flat_accuracy_score(y_true, y_pred)

    print("accuracy:",accuracy)
    print("f1:",f1)
    print("precision:",f1)
    print("recall:",recall)


    import plotly
    import plotly.graph_objects as go

    flat_y_true=[]
    flat_y_pred=[]

    for x in y_true:
        for y in x:
            flat_y_true.append(y)

    for x in y_pred:
        for y in x:
            flat_y_pred.append(y)

    end_p=["RP","NFP","VBP","NNP","PRP","WP"]
    for i in range(0,len(flat_y_true)):
        if flat_y_true[i][-1]=="P" and flat_y_true[i][-1] not in end_p:
            flat_y_true[i]="PUNCT"
        if flat_y_pred[i][-1]=="P" and flat_y_pred[i][-1] not in end_p:
            flat_y_pred[i]="PUNCT"

    #print(type(flat_y_true))
    #print(flat_y_true[0],flat_y_true[-1])



def save(filename):	#filename shall end with .pickle and type(filename)=string
    print("Saving classifier.")
    with open(filename, "wb") as f:
        pickle.dump(classifier, f)
    return


def load(filename):	#filename shall end with .pickle and type(filename)=string
    print("Loading classifier...")
    print(filename)
    with open(filename, "rb") as f:
        classifier=pickle.load(f)
        return classifier




def tag(sentence):
    #takes a single sentence as a list
    classifier=load("pos_crf.pickle")
    t_features=[]
    for j in range(0,len(sentence)):
        t_features.append(feature_extractor(sentence,j))

    #print(sentence)
    #print(len(t_features))

    ret=classifier.predict([t_features])[0]
    end_p=["RP","NFP","VBP","NNP","PRP","WP"]
    for i in range(0,len(ret)):
        if ret[i][-1]=="P" and ret[i][-1] not in end_p:
            ret[i]="PUNCT"

    return ret



if __name__ == "__main__":
	stemmer = SnowballStemmer("english")
	english_stopwords = set(stopwords.words('english'))
	classifier=sklearn_crfsuite.CRF(c1=0.2, c2=0.2, max_iterations=1000)
	training_data, test_data=creatsets("en-ud-train.conllu")

	with open('pos_crf_train.data', 'rb') as file:
		training_data=pickle.load(file)
	file.close()
	


	train(training_data)
	save("pos_crf.pickle")


	with open('pos_crf_test.data', 'rb') as file:
		test_data=pickle.load(file)
	file.close()

	classifier=load("pos_crf.pickle")
	test(test_data)

	s=['The',
    'guitarist',
    'died',
    'of',
    'a',
    'drugs',
    'overdose',
    'in',
    '1970',
    'aged',
    '27',
    '.']

	print(tag(s))