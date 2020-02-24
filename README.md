# BigBert

This assignment is part of the Natural Language Processing module of Imperial 
College London. The task is the implementation of a regression model that predicts 
the quality of a machine translated sentence between English and German. 
This task will be referred to as Machine Translation Evaluation (MTE).  

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

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

### Embeddings Example 

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
embeddings_max, embeddings_mean = english_encoder.get_batch_embedding(
                                data, batch_size = batch_size)
#Processing batch 1...
#Getting embedding for batch 1...

#Processing batch 2...
#Getting embedding for batch 2...

#DONE!
#Embedding size is (10,768)

```
### Load Saved Embeddings

```

```



## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)


