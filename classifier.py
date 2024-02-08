from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from gensim.models import Word2Vec, KeyedVectors

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from datasets import Dataset
import evaluate
from tqdm.auto import tqdm
import numpy as np

import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RandomForest():
    def __init__(self,training_data, testing_data, n_estimators=None, max_depth=None, tfidf=False):
        self.training_data = training_data['text'].tolist()
        self.train_labels = training_data['labels'].tolist()
        self.testing_data = testing_data['text'].tolist()
        self.test_labels = testing_data['labels'].tolist()

        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.max_len = None
        self.train_input = []
        self.val_input = None
        self.val_labels = None
        self.test_input = []
        self.classifier = None
        self.vectorizer = None

        self.best_f1 = None
        self.best_model = None

        self.max_len = len(max([sentence.split() for sentence in self.training_data], key=len))

        if tfidf:
            print("Generating Tf-IDF Vectors...")
            self.getTfidfvectors(self.training_data)
            self.getTfidfvectors(self.testing_data, istest=True)
        else:
            print("Generating word embeddings...")
            self.getFeatures(self.training_data, self.train_input)
            self.getFeatures(self.testing_data, self.test_input)

    def getFeatures(self, data, embeddings):
        wv_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

        for sentence in data:
            sentence = sentence.split()
            word2idx = []
            for ind in range(len(sentence)):
                if wv_model.has_index_for(sentence[ind]):
                    word2idx.append(wv_model[sentence[ind]])

            combined_vecs = np.zeros([300])
            for vector in word2idx:
                combined_vecs = np.add(combined_vecs,vector)
            embeddings.append(combined_vecs)

    def getTfidfvectors(self, data, istest=False):
        if not istest:
            vectorizer = TfidfVectorizer()
            self.vectorizer = vectorizer.fit(data)
            self.train_input = self.vectorizer.transform(data)
        else:
            self.test_input = self.vectorizer.transform(data)


    def train(self):
        self.classifier = RandomForestClassifier()
        parameters = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth
        }

        print('Training model..')
        cv = GridSearchCV(self.classifier,parameters)
        cv.fit(self.train_input,self.train_labels)

        self.best_model = cv.best_estimator_
        print(self.best_model)

    def predict(self, valid=False):
        if valid:
            test_predictions = self.best_model.predict(self.val_input)
        else:
            print("Testing model...")
            test_predictions = self.best_model.predict(self.test_input)
        return test_predictions


class Transformer():
    def __init__(self, training_data, testing_data, model_name=None):
        self.label2id = {'human': 0, 'llama': 1, 'falcon': 2, 'mistral': 3}
        self.id2label = {0: 'human', 1: 'llama', 2: 'falcon', 3: 'mistral'}

        self.training_data = training_data
        self.training_data['labels'] = self.training_data['labels'].apply(lambda x: self.label2id[x])
        self.testing_data = testing_data
        self.testing_data['labels'] = self.testing_data['labels'].apply(lambda x: self.label2id[x])
        self.val_data = None

        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        self.best_f1 = None
        self.best_model = None
        self.best_epoch = None

    def preprocess(self,examples):
        return self.tokenizer(examples["text"],max_length=512,padding='max_length')

    def train(self):
        # load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name,id2label=self.id2label, num_labels=len(self.label2id), lable2id=self.label2id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)

        # split data into train and validation
        self.training_data, self.val_data = train_test_split(self.training_data,test_size=0.2,stratify=self.training_data['labels'],random_state=256)

        training_dataset = Dataset.from_pandas(self.training_data)
        valid_dataset = Dataset.from_pandas(self.val_data)

        # tokenize the inputs
        training_dataset = training_dataset.map(self.preprocess, batched=True)
        valid_dataset = valid_dataset.map(self.preprocess, batched=True)
        training_dataset = training_dataset.remove_columns(['text','__index_level_0__'])
        valid_dataset = valid_dataset.remove_columns(['text','__index_level_0__'])
        training_dataset.set_format('torch')
        valid_dataset.set_format('torch')

        # create dataloaders with batch size
        training_dataloader = DataLoader(training_dataset, batch_size=4, shuffle=False)
        val_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

        # set parameters
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        num_epochs = 3
        num_training_steps = num_epochs * len(training_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        self.model.to(device)

        progress_bar = tqdm(range(num_training_steps))

        # train the model
        self.model.train()
        for epoch in range(num_epochs):
            for batch in training_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # evaluate the model on validation data
            f1 = self.predict(val_dataloader)['f1']

            if self.best_f1 is None or self.best_f1 < f1:
                self.best_model = self.model
                self.best_f1 = f1
                self.best_epoch = epoch
        
        # print('The best model F1-score was {} at epoch {}'.format(self.best_f1,self.best_epoch))

    def predict(self, data=None):
        test=False
        if not data:
            model = self.best_model
            testing_dataset = Dataset.from_pandas(self.testing_data)
            testing_dataset = testing_dataset.map(self.preprocess, batched=True)
            testing_dataset = testing_dataset.remove_columns(['text'])
            testing_dataset.set_format('torch')
            data = DataLoader(testing_dataset, batch_size=16, shuffle=False)
            test=True
        else:
            model = self.model

        model.eval()
        metric_1 = evaluate.load("f1")
        predicted_labels = []
        gold_labels = []

        for batch in tqdm(data):
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
            except:
                print(batch)
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            if test:
                predicted_labels+=predictions.tolist()
                gold_labels+=batch['labels'].tolist()
            else:
                metric_1.add_batch(predictions=predictions, references=batch["labels"])

        if test:
            return classification_report(gold_labels,predicted_labels,digits=3)
        else:
            return metric_1.compute(average="macro")