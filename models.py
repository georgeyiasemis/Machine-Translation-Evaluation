from scipy.stats.stats import pearsonr
from sklearn.neural_network import MLPRegressor
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVR

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

class MLP_Regressor():
    
    def __init__(self, activation='relu', regularization = 0.005, batch_size=128,
             hidden_layer_sizes=(4096, 2048, 1024,512, 256, 128), learning_rate='adaptive',
             learning_rate_init=0.001, max_iter=25, n_iter_no_change=10,
             optimizer='adam', early_stopping=True, tol=0.0001, validation_fraction=0.15):
        
        self.model = MLPRegressor(activation=activation, alpha=regularization, 
                                  batch_size=batch_size,
                                  hidden_layer_sizes=hidden_layer_sizes, 
                                  learning_rate=learning_rate,
                                  learning_rate_init=learning_rate_init, 
                                  max_iter=max_iter,
                                  n_iter_no_change=n_iter_no_change, solver=optimizer, 
                                  early_stopping=early_stopping,
                                  tol=tol, validation_fraction=validation_fraction,
                                  verbose=True)
        
    def fit(self, x, y):
        print('Training...')
        self.model.fit(x, y)
    
    def predict(self, x_test, y_test=None):
        
        y_pred = self.model.predict(x_test)
        
        if y_test != None:
            return y_pred
        else:
            pearson, rmse_score = self.get_scores(y_test, y_pred)
            return y_pred, pearson, rmse_score
            
    
    def get_scores(self,  y_test, y_pred):
        pearson = pearsonr(y_test, y_pred)[0]
        rmse_score = rmse(y_pred, y_test)
        
        return pearson, rmse_score
        
class MLP(nn.Module):

    def __init__(self, layers_sizes):
        super(MLP, self).__init__()
        self.layers_sizes = layers_sizes
        self.network = nn.Sequential()
        for i, (in_dims, out_dims) in enumerate(zip(self.layers_sizes[:-1], self.layers_sizes[1:])):
            self.network.add_module(
                name=f'Linear {i}',module=nn.Linear(in_dims, out_dims))
            if i != len(self.layers_sizes) - 2:
                self.network.add_module(name=f'Activation {i}', module=nn.ReLU())
                self.network.add_module(name=f'Dropout {i}', module=nn.Dropout(0.4))
            else:
                self.network.add_module(name='Identity', module=nn.Identity())
            
    def forward(self, x):
        
        return self.network(x)
    

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

    

    def fit(self):
        print('Training...')
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