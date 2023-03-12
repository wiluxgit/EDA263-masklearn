# %% [markdown]
# # Task 1: Working with a dataset with categorical features

# %% [markdown]
# ### Step 1, Reading the data

# %%
import pandas as pd
import numpy as np
import sklearn as sk
import krippendorff
import torch 

class TDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    
class DataSetContainer():
    def __init__(self, RawY=None, RawX=None, File=None, Consensus=None, SplitY=None, ConfidenceWeights=None, TorchDataSet=None):
        self.RawX = RawX
        self.RawY = RawY
        self.SplitY = SplitY
        self.File = File
        self.Consensus = Consensus
        self.ConfidenceWeights = ConfidenceWeights

# %%
trainSet = DataSetContainer(File="assets/a3_train_final.tsv")
testSet = DataSetContainer()
dataContainers = [trainSet, testSet]

for dcont in [trainSet]:
    df = pd.read_table(dcont.File, names=['opinion', 'text'])
    df = df.sample(frac=1, random_state=1337)
    df["text"] = df["text"].apply(lambda a:a.lower())
    dcont.RawY = df["opinion"]
    dcont.RawX = df["text"]

# %%
# The trainset has annotator disagreements
# https://towardsdatascience.com/assessing-annotator-disagreements-in-python-to-build-a-robust-dataset-for-machine-learning-16c74b49f043

import numpy as np

import random
from collections import Counter
allNums = trainSet.RawY.str.split('/').to_numpy()
allNums = np.concatenate(allNums).ravel().tolist()
occNum = dict(Counter(allNums))
del occNum['-1']

def randomByOcc():
    val,prob = zip(*(occNum.items()))
    #return int(random.choices([0,1], weights=[1,9]))
    return int(random.choices(val,weights=prob))

def toNumOrNan(n):
    try:
        if (n == "-1"):
            #return randomByOcc()
            #return 0
            return np.nan
        return int(n)
    except Exception:
        return np.nan

for dset in [trainSet]:
    splitOpinion = dset.RawY.str.split('/', expand=True)
    splitOpinion = splitOpinion.applymap(toNumOrNan).transpose()
    
    # since we don't know who the annotators are who wrote what should be arbitrary
    # but (it does not actually matter for krippendorf)
    #splitOpinion = pd.DataFrame(data=[sk.utils.shuffle(list(splitOpinion.loc[:,c]), random_state=c) for c in splitOpinion.columns]).transpose()

    dset.SplitY = splitOpinion
    dset.Consensus = krippendorff.alpha(reliability_data=splitOpinion, value_domain=[0,1])
print(f"Krippendorff alpha for training data: {trainSet.Consensus}")

# %%
############
# Bert Setup
############
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased'
)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# %%
# Weiging annotations, https://arxiv.org/pdf/2208.06161.pdf
#   SPA makes one key assumption: The degree to
#   which labels are absent must be independent of the
#   true item-agreements ni⊥Pi.
from collections import Counter

def getMostLikelyAndItsWeight(col):
    answer2count = Counter([x for x in col if x in [0,1]])
    nAnnotators = float(len(answer2count))

    mostPopularAnswer = sorted(answer2count, reverse=True)[0]
    mostPopularCount = answer2count[mostPopularAnswer]

    # agreement = % is the most popular - % isn't the most popular
    del answer2count[mostPopularAnswer]
    agreement = float(mostPopularCount - sum(answer2count.values()))/nAnnotators

    #using weight = number of annotators
    weight = nAnnotators

    return (weight*agreement, mostPopularAnswer)

train_weights,train_mostPopClass = zip(*[
    getMostLikelyAndItsWeight(dset.SplitY.loc[:,c]) 
    for c in trainSet.SplitY.columns
])
trainSet.ConfidenceWeights = pd.Series(list(train_weights))

# %%
# max tweet = 240 characters -> 120 words
# must const sizes or torch gets angry
from sklearn.model_selection import train_test_split 
X_train, X_val, y_train, y_val = train_test_split(trainSet.RawX, train_mostPopClass, test_size=0.2)

train_Xtoken = tokenizer(list(X_train), max_length=120, padding=True, truncation=True) 
eval_Xtoken  = tokenizer(list(X_val),  max_length=120, padding=True, truncation=True)

# %%
trainY = [int(x) for x in list(y_train)]
evalY = [int(x) for x in list(y_val)]

trainTset = TDataset(train_Xtoken, trainY)
evalTset = TDataset(eval_Xtoken, evalY)

#print(list(train_Xtoken)[:10])
print(len(train_Xtoken["input_ids"]))
print(len(train_Xtoken["token_type_ids"]))
print(len(train_Xtoken["attention_mask"]))
print(len(trainY))

print(trainY)
#display(trainX.head())

# %%
def getScores(torchDataSet):
    pred, labels = torchDataSet
    pred = np.argmax(pred, axis=1)

    recall = sk.metrics.recall_score(y_true=labels, y_pred=pred)
    precision = sk.metrics.precision_score(y_true=labels, y_pred=pred)
    f1 = sk.metrics.f1_score(y_true=labels, y_pred=pred)

    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1
    }

# %%
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

#display(trainSet.TorchDataSet)

#https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=1337,
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainTset,
    eval_dataset=evalTset,
    compute_metrics=getScores,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# %%
import os
os.environ["WANDB_DISABLED"] = "true"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1"

# %%
trainer.train()

# %% [markdown]
# # Testing

# %%
model_path = "output/checkpoint-50000"
#model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)


