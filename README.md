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

### Example

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

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

