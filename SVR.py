from sklearn.svm import SVR
from scipy.stats.stats import pearsonr
import io
import torch
import numpy as np
from zipfile import ZipFile

from embedding import Embedding

class SVR_regression():
    
    def __init__(self, c=0.1, epsilon=0.1, kernel='rbf', embedding_mode=3):
        # Default init attributes is the optimal hyperparameters
        self.c=c
        self.epsilon=epsilon
        self.kernel=kernel
        self.embedding_mode=embedding_mode

        self.X_train = None
        self.X_val = None
        self.train_scores = None
        self.val_scores = None

        self.svr = None

    def load_embeddings(self):

        with io.open("./data_en_de/dev.ende.scores", "r", encoding="utf8") as ende_mt:
            german_val_scores = ende_mt.read().splitlines()
        with io.open("./data_en_de/train.ende.scores", "r", encoding="utf8") as ende_mt:
            german_train_scores = ende_mt.read().splitlines()
        with io.open("./data_en_de/test.ende.mt") as ende_mt:
            test_set = ende_mt.read().splitlines()
            self.X_test = np.array(test_set)

        #Scores
        self.train_scores = np.array(german_train_scores).astype(float)

        self.val_scores = np.array(german_val_scores).astype(float)

        if self.embedding_mode == 1:
            # Basic BERT embeddings (N*768)
            de_embeddings = torch.load('./Embeddings/german_embeddings.pt').numpy()
            de_val_embeddings = torch.load('./Embeddings/german_val_embeddings.pt').numpy()
            en_embeddings = torch.load('./Embeddings/english_embeddings.pt').numpy()
            en_val_embeddings = torch.load('./Embeddings/english_val_embeddings.pt').numpy()

            self.X_train= np.concatenate((en_embeddings,de_embeddings), axis=1) #concatenate english and german sentence vectors
            self.X_val = np.concatenate((en_val_embeddings,de_val_embeddings), axis=1)
        elif self.embedding_mode == 2:
            # with lemmatization, dont remove stop words
            de_embeddings_max = torch.load('./Embeddings\ with\ not\ removed\ stopwords\ -\ max\ pooling/german_embeddings_max.pt').numpy()
            de_embeddings_mean = torch.load('./Embeddings\ with\ not\ removed\ stopwords\ -\ avg\ pooling/german_embeddings.pt').numpy()
            de_val_embeddings_max = torch.load('./Embeddings\ with\ not\ removed\ stopwords\ -\ max\ pooling/german_val_embeddings_max.pt').numpy()
            de_val_embeddings_mean = torch.load('./Embeddings\ with\ not\ removed\ stopwords\ -\ avg\ pooling/german_val_embeddings.pt').numpy()
            en_embeddings_max = torch.load('./Embeddings\ with\ not\ removed\ stopwords\ -\ max\ pooling/english_train_embeddings_max.pt').numpy()
            en_embeddings_mean = torch.load('./Embeddings\ with\ not\ removed\ stopwords\ -\ avg\ pooling/english_train_embeddings.pt').numpy()
            en_val_embeddings_max = torch.load('./Embeddings\ with\ not\ removed\ stopwords\ -\ max\ pooling/english_val_embeddings_max.pt').numpy()
            en_val_embeddings_mean = torch.load('./Embeddings\ with\ not\ removed\ stopwords\ -\ avg\ pooling/english_val_embeddings.pt').numpy()

            en_train= np.concatenate((en_embeddings_max, en_embeddings_mean), axis=1) #concatenate english and german sentence vectors
            de_train= np.concatenate((de_embeddings_max, de_embeddings_mean), axis=1) #concatenate english and german sentence vectors
            x_train_abs = abs(en_train-de_train) #calculate absolute difference
            x_train_prod = np.multiply(en_train, de_train) #multiply elementwise
            #concatenate all embedding vectors together
            X_train = np.concatenate((en_train, de_train), axis=1)
            X_train = np.concatenate((X_train, x_train_abs), axis=1)
            X_train = np.concatenate((X_train, x_train_prod), axis=1)
            self.X_train = X_train

            en_val= np.concatenate((en_val_embeddings_max, en_val_embeddings_mean), axis=1) #concatenate english and german sentence vectors
            de_val= np.concatenate((de_val_embeddings_max, de_val_embeddings_mean), axis=1) #concatenate english and german sentence vectors
            x_val_abs = abs(en_val-de_val)
            x_val_prod = np.multiply(en_val, de_val)
            X_val = np.concatenate((en_val, de_val), axis=1)
            X_val = np.concatenate((X_val, x_val_abs), axis=1)
            X_val = np.concatenate((X_val, x_val_prod), axis=1)
            self.X_val = X_val

        elif self.embedding_mode == 3:
            # no lemmatization, dont remove stop words
            de_embeddings_max = torch.load('./Emb_no_lem_no_stop_words_removed/german_embeddings_max.pt').numpy()
            de_embeddings_mean = torch.load('./Emb_no_lem_no_stop_words_removed/german_embeddings_mean.pt').numpy()
            de_val_embeddings_max = torch.load('./Emb_no_lem_no_stop_words_removed/german_val_embeddings_max.pt').numpy()
            de_val_embeddings_mean = torch.load('./Emb_no_lem_no_stop_words_removed/german_val_embeddings_mean.pt').numpy()
            en_embeddings_max = torch.load('./Emb_no_lem_no_stop_words_removed/english_train_embeddings_max.pt').numpy()
            en_embeddings_mean = torch.load('./Emb_no_lem_no_stop_words_removed/english_train_embeddings_mean.pt').numpy()
            en_val_embeddings_max = torch.load('./Emb_no_lem_no_stop_words_removed/english_val_embeddings_max.pt').numpy()
            en_val_embeddings_mean = torch.load('./Emb_no_lem_no_stop_words_removed/english_val_embeddings_mean.pt').numpy()

            en_train= np.concatenate((en_embeddings_max, en_embeddings_mean), axis=1) #concatenate english and german sentence vectors
            de_train= np.concatenate((de_embeddings_max, de_embeddings_mean), axis=1) #concatenate english and german sentence vectors
            x_train_abs = abs(en_train-de_train)
            x_train_prod = np.multiply(en_train, de_train)
            X_train = np.concatenate((en_train, de_train), axis=1)
            X_train = np.concatenate((X_train, x_train_abs), axis=1)
            X_train = np.concatenate((X_train, x_train_prod), axis=1)
            self.X_train = X_train

            en_val= np.concatenate((en_val_embeddings_max, en_val_embeddings_mean), axis=1) #concatenate english and german sentence vectors
            de_val= np.concatenate((de_val_embeddings_max, de_val_embeddings_mean), axis=1) #concatenate english and german sentence vectors
            x_val_abs = abs(en_val-de_val)
            x_val_prod = np.multiply(en_val, de_val)
            X_val = np.concatenate((en_val, de_val), axis=1)
            X_val = np.concatenate((X_val, x_val_abs), axis=1)
            X_val = np.concatenate((X_val, x_val_prod), axis=1)
            self.X_val = X_val


    def fit(self):

        self.svr = SVR(kernel = self.kernel, C=self.c, epsilon=self.epsilon, verbose=True)
        self.svr.fit(self.X_train, self.train_scores)

    def predict(self, set='val'):
        #Predicts
        if set == 'val':
            predictions = self.svr.predict(self.X_val)
        elif set == 'test':
            predictions = self.svr.predict(self.X_test)
        
        pearson = pearsonr(self.val_scores, predictions)
        RMSE = np.sqrt(((predictions - self.val_scores) ** 2).mean())
        
        print(f'RMSE: {RMSE} Pearson {pearson[0]}')
        print()

        return predictions

    def writeScores(self, scores):
        #from baseline
        fn = "predictions.txt"
        print("")
        with open(fn, 'w') as output_file:
            for idx,x in enumerate(scores):
                output_file.write(f"{x}\n")

    def run_model(self):
        #runs model and pickles output
        self.fit()
        predictions = self.predict(set='test')

        self.writeScores(predictions)

        with ZipFile("en-de_svr.zip","w") as newzip:
            newzip.write("predictions.txt")

    def gridsearch(self):
        '''
        Rudimentary implementation of grid search. Prints dictionary of pearson and RMSE scores 
        for various combinations of hyperparameters
        '''

        outputs = dict()

        for C in [0.1, 1, 10]:
            for e in [0.01, 0.1, 1]:
                for k in ['linear', 'poly','rbf','sigmoid']:
                    reg = SVR(kernel=k, verbose=True, C=C, epsilon=e)
                    reg.fit(self.X_train, self.train_scores)
                    predictions = reg.predict(self.X_val)
                    pearson = pearsonr(self.val_scores, predictions)
                    RMSE = np.sqrt(((predictions - self.val_scores) ** 2).mean())
                    pearson = pearson[0]
                    stats = (RMSE, pearson)
                    keyname = str(C) + '/' + str(e) + '/' + k
                    outputs[keyname] = stats
                    print(outputs)
                    print(f'RMSE: {RMSE} Pearson {pearson}')
                    print()

        print(outputs)

if __name__ == "__main__":
    regressor = SVR_regression(c=0.1, epsilon=0.01, kernel='rbf')
    regressor.load_embeddings()
    regressor.run_model()
    #regressor.gridsearch()




'''
EXPERIMENT RESULTS BELOW


Grid Search Results, embedding_mode = 3:

{'0.1/0.01/linear': (0.897847740175238, 0.05393922813245665),
 '0.1/0.01/poly': (0.8747754342014558, 0.1297285406861192),
 '0.1/0.01/rbf': (0.8751126608340087, 0.13221871955081785),
  '0.1/0.01/sigmoid': (0.8770704796428923, 0.11848639234963391),
  '0.1/0.1/linear': (0.8910136425341122, 0.07157336128353219),
   '0.1/0.1/poly': (0.8756428145346494, 0.10793126515408055),
   '0.1/0.1/rbf': (0.8759593814602268, 0.10767879230359444),
    '0.1/0.1/sigmoid': (0.877045630219154, 0.10623146419359862),
    '1/0.01/linear': (0.9053442047442838, 0.04409849375313772),
     '1/0.01/poly': (0.8779323112443164, 0.0931037404146863),
     '1/0.01/rbf': (0.8759328288148905, 0.10687957383411816),
      '1/0.01/sigmoid': (0.8793275314452126, 0.08068017586303596),
      '1/0.1/linear': (0.8980087968767988, 0.06234251442043081),
       '1/0.1/poly': (0.8771841561370594, 0.08862143881915104),
       '1/0.1/rbf': (0.8761773377440072, 0.09708708914876642),
        '1/0.1/sigmoid': (0.877745714295036, 0.08550917288366752),
        '1/1/linear': (1.0532353115219217, 0.02322739955909181),
         '1/1/poly': (0.9138833397209004, 0.10083704918828976),
         '1/1/rbf': (0.9083508506737453, 0.10954171268993163),
          '1/1/sigmoid': (0.8978778668186665, 0.11690439712027718),
          '10/0.01/linear': (0.9074498445618453, 0.03874582222523645),
            '10/0.01/poly': (0.9016781531764824, 0.05781440253772647),
             '10/0.01/rbf': (0.8917040524434375, 0.06385628636723463),
              '10/0.01/sigmoid': (2.8867472629223863, 0.04128175717587749),
                '10/0.1/poly': (0.9627359678433621, 0.05153409683255006),
                '10/0.1/rbf': (0.9097816639113756, 0.04628297394363309),
                 '10/0.1/sigmoid': (10.720972378575919, 0.02105002783422333),
                 '10/1/poly': (1.023877798689137, 0.06693690106195421),
                 '10/1/rbf': (0.967363612198251, 0.08254992819125392),
                 '10/1/sigmoid': (10.636160602296705, 0.02174298746575983)}
'''


'''
Default settings - ENCODING MODE 1 (BERT EMBEDDINGS RAW)

linear
RMSE: 0.8980087968767988 Pearson 0.062342514420430824

poly
RMSE: 0.8774419328872658 Pearson 0.11577634632380732

rbf
RMSE: 0.8750561970401898 Pearson 0.10917615455766054

sigmoid
RMSE: 0.8759619775776275 Pearson 0.1023711322652818


C = 0.1
linear
RMSE: 0.8910136425341122 Pearson 0.07157336128353219
poly
RMSE: 0.8804698140048252 Pearson 0.1229665445843766
rbf
RMSE: 0.877111306253335 Pearson 0.11078279074483584
sigmoid
RMSE: 0.8776929748238558 Pearson 0.11837061026434267

C=0.1, e = 0.01
[LibSVM]linear
RMSE: 0.897847740175238 Pearson 0.05393922813245665
[LibSVM]poly
RMSE: 0.8747754342014558 Pearson 0.1297285406861192
[LibSVM]rbf
RMSE: 0.8751126608340087 Pearson 0.13221871955081785
[LibSVM]sigmoid
RMSE: 0.8770704796428923 Pearson 0.11848639234963391

C=0.1,e=0.1
[LibSVM]linear
RMSE: 0.8910136425341122 Pearson 0.07157336128353219
[LibSVM]poly
RMSE: 0.8756428145346494 Pearson 0.10793126515408055
[LibSVM]rbf
RMSE: 0.8759593814602268 Pearson 0.10767879230359444
[LibSVM]sigmoid
RMSE: 0.877045630219154 Pearson 0.10623146419359862

C=0.1,e=1
[LibSVM]linear
RMSE: 0.9649254062706102 Pearson 0.03622344520992335
[LibSVM]poly
RMSE: 0.8985691663631685 Pearson 0.14552392013419113
[LibSVM]rbf
RMSE: 0.8990307590698758 Pearson 0.14950919041942473
[LibSVM]sigmoid
RMSE: 0.901106856235984 Pearson 0.15769695985426876

C = 1, e=0.01
[LibSVM]linear
RMSE: 0.9053442047442838 Pearson 0.04409849375313772
[LibSVM]poly
RMSE: 0.8779323112443164 Pearson 0.0931037404146863
[LibSVM]rbf
RMSE: 0.8759328288148905 Pearson 0.10687957383411816
[LibSVM]sigmoid
RMSE: 0.8793275314452126 Pearson 0.08068017586303596



'''