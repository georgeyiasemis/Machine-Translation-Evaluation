# BigBert

This assignment is part of the Natural Language Processing module of Imperial 
College London. The task is the implementation of a regression model that predicts 
the quality of a machine translated sentence between English and German. 
This task will be referred to as Machine Translation Evaluation (MTE).  

## Getting Started

These instructions will get you a copy of the project up and running on your local 
machine for development and testing purposes. See deployment for notes on how to deploy
the project on a live system.

#Run pipeline

Below are intructions of how to run the pipeline (we do not create the embeddings from scratch in this format. If you want to create the embeddings see the Get Embeddings format):
Download the whole directory 
Install the Prerequisites
ru
 

### Prerequisites

What things you need to install the software and how to install them

```
transformers
sklearn
scipy
torch
numpy
nltk
```

### Get Embeddings Example 

How to get embeddings for:
```
data = ['You can say to me.',
        "I love machine learning It's awesome",
        "I love coding in python",           
        "I love building chatbots",
        "they chat amagingly well",
        'Hello Mr. Smith, how are you doing today?', 
        'The weather is great, and city is awesome.', 
        'The sky is pinkish-blue.', 
        "You shouldn't eat cardboard",
        'Hello beautiful world']
print(len(data))

#10
```
Create an encoder:

```
import Embedding from embedding

english_encoder = Embedding('en')
#Loading English BERT tokenizer...

#Loading English BERT Model...
```

Get the embeddings:

```
batch_size = 5
lemmatize = True
remove_stop_words = False

embeddings_max, embeddings_mean = english_encoder.get_batch_embedding(
                                data, batch_size = batch_size,lemmatize=lemmatize, 
                                remove_stop_words=remove_stop_words)
#Processing batch 1...
#Getting embedding for batch 1...

#Processing batch 2...
#Getting embedding for batch 2...

#DONE!
#Embedding size is (10,768)

```
### Load Saved Embeddings Example

```
import torch

# Load Embeddings

english_max = torch.load('./Embeddings/english_train_embeddings_max.pt')
german_max = torch.load('./Embeddings/german_embeddings_max.pt')

english_avg = torch.load('./Embeddings/english_train_embeddings_mean.pt')
german_avg = torch.load('./Embeddings/german_embeddings_mean.pt')

# Load scores/labels

f_train_scores = open("./data_en_de/train.ende.scores",'r')
de_train_scores = f_train_scores.readlines()

```
## Create feature vectors
```
y_train = np.array(de_train_scores).astype(float)
# Shape (7000,)


# Tensor of Shape (7000, 768 x 2)
english = torch.cat((english_max, english_avg), dim=1)
german = torch.cat((german_max, german_avg), dim=1)

en_ge_cat = torch.cat((english,german), dim=1)
en_ge_product = english * german
en_ge_abs_dif = (english - german).abs()

# Tensor of Shape (7000, 768 x 8)
X_train = torch.cat((en_ge_cat, en_ge_product, en_ge_abs_dif), dim=1)

```

### Train a model

Choose a mode from {```'SVR', 'MLP_torch', 'MLP_sckit'```} and execute ```main.py```.
You can define all the hyperparameters in the ```main()``` fucntion.

## Authors

* **George Yiasemis** - *CID:11008587*
* **Christos Seas** - *CID:01836251*
* **Dhruva Storz** - *CID:01807283*


