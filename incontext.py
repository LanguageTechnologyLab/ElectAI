from langchain import PromptTemplate
from transformers import AutoTokenizer, pipeline
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import re
import sys
from transformers.utils import logging
logging.set_verbosity_error()


data = pd.read_csv('./datasets/claim understanding/claims-annotated.csv',sep='\t')
columns = list(data.columns)

task = "text-generation"
if sys.argv[1].lower() == "llama":
    model = "meta-llama/Llama-2-7b-chat-hf"
elif sys.argv[1].lower() == "falcon":
    model = "tiiuae/falcon-7b-instruct"
elif sys.argv[1].lower() == "mistral":
    model = "mistralai/Mistral-7B-Instruct-v0.2"
elif sys.argv[1].lower() == "flan":
    model = "google/flan-t5-xl"
    task = "text2text-generation"
else:
    print("ERROR: Invalid model name.")

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    task=task, #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype="auto",
    device_map="auto",
    max_new_tokens=1,
    do_sample=True,
    top_k=1,
    top_p=0.95,
    temperature=0.95
)


template = """Tweet: {tweet}
Answer the following question with a yes or a no.
Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["tweet","question"])

questions = ['Does the tweet mention a jurisdiction?',
             'Does the tweet mention a state election?',
             'Does the tweet mention a county election?',
             'Does the tweet mention federal election?',
             'Does the tweet mention any election equipment?',
             'Does the tweet mention electronic voting equipment machines?',
             'Does the tweet mention ballots or related equipment?',
             'Does the tweet mention any election-related process?',
             'Does the tweet mention the vote counting process?',
             'Does the tweet mention any election-related claim of fraud?',
             'Does the tweet mention corruption in elections?',
             'Does the tweet mention illegal voting?',
            ]

predicted_labels = []
gold_labels = []

for ind, row in tqdm(data.iterrows(),total=data.shape[0]):
    prediction = []
    for question in questions:
        response = pipeline(prompt.format(tweet=row['tweet'],question=question))[0]['generated_text']
        if model != "google/flan-t5-xl":
            span = re.finditer('Answer:',response)
            span = max(enumerate(span))[1]
            response = response[span.end()+1:]
        response = response.lower()
        prediction.append(1 if response=="yes" else 0)
    predicted_labels.append(prediction)
    gold = list(row[columns[1:]])
    gold_labels.append(gold)

print(classification_report(gold_labels,predicted_labels,target_names=columns[1:]))