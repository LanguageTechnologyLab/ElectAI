import pandas as pd
from classifier import Transformer, RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from cleantext import replace_urls
import re
import sys


train_data = pd.read_csv('./datasets/authorship attribution/train.csv', sep='\t').sample(frac=1,random_state=1762)

# preprocess the train set
train_data = train_data.rename(columns={'tweet':'text', 'label':'labels'})
train_data['text'] = train_data['text'].apply(lambda x: x.strip())
train_data['text'] = train_data['text'].apply(lambda x: re.sub('@[a-zA-Z0-9\_]+ ','USER ',x))
train_data['text'] = train_data['text'].apply(lambda x: replace_urls(x,replace_with='URL'))

test_data = pd.read_csv('./datasets/authorship attribution/test.csv', sep='\t')

# preprocess the test set
test_data = test_data.rename(columns={'tweet':'text', 'label':'labels'})
test_data['text'] = test_data['text'].apply(lambda x: x.strip())
test_data['text'] = test_data['text'].apply(lambda x: re.sub('@[a-zA-Z0-9\_]+ ','USER ',x))
test_data['text'] = test_data['text'].apply(lambda x: replace_urls(x,replace_with='URL'))

if sys.argv[1].lower() == "bert":
    model = Transformer(train_data,test_data,"bert-base-uncased")
    model.train()
    class_report = model.predict()
    print(class_report)
    print()
elif sys.argv[1].lower() == "roberta":
    model = Transformer(train_data,test_data,"FacebookAI/roberta-base")
    model.train()
    class_report = model.predict()
    print(class_report)
    print()
elif sys.argv[1].lower() == "rf" and sys.argv[2].lower() == "word2vec":
    model = RandomForest(train_data,test_data,[50,100,200],[20,40,50])
    model.train()
    predictions = model.predict()
    print(classification_report(model.test_labels,predictions,digits=3))
    print()
elif sys.argv[1].lower() == "rf" and sys.argv[2].lower() == "tfidf":
    model = RandomForest(train_data,test_data,[50,100,200],[20,40,50],tfidf=True)
    model.train()
    predictions = model.predict()
    print(classification_report(model.test_labels,predictions,digits=3))
else:
    print("ERROR: Invalid model or feature name.")