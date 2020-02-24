import torch
import numpy as np
import models
# Max pooling
english_m = torch.load('./Embeddings/english_train_embeddings_max.pt')
german_m = torch.load('./Embeddings/german_embeddings_max.pt')
english_val_m = torch.load('./Embeddings/english_val_embeddings_max.pt')
german_val_m = torch.load('./Embeddings/german_val_embeddings_max.pt')
english_test_m = torch.load('./Embeddings/english_test_embeddings_max.pt')
german_test_m = torch.load('./Embeddings/german_test_embeddings_max.pt')
# Avg pooling
english_avg = torch.load('./Embeddings/english_train_embeddings_mean.pt')
german_avg = torch.load('./Embeddings/german_embeddings_mean.pt')
english_val_avg = torch.load('./Embeddings/english_val_embeddings_mean.pt')
german_val_avg = torch.load('./Embeddings/german_val_embeddings_mean.pt')
english_test_avg = torch.load('./Embeddings/english_test_embeddings_mean.pt')
german_test_avg = torch.load('./Embeddings/german_test_embeddings_mean.pt')

# LOAD Scores

f_train_scores = open("/content/gdrive/My Drive/train.ende.scores",'r')
de_train_scores = f_train_scores.readlines()
f_val_scores = open("/content/gdrive/My Drive/dev.ende.scores",'r')
de_val_scores = f_val_scores.readlines()

train_scores = np.array(de_train_scores).astype(float)
# Shape (7000,)
y_train =train_scores
# Shape (1000,)
val_scores = np.array(de_val_scores).astype(float)
y_val =val_scores

# Shape (7000, 768 x 2)
english = torch.cat((english_m, english_avg), dim=1)
german = torch.cat((german_m, german_avg), dim=1)

# Shape (1000, 768 x 2)
english_val = torch.cat((english_val_m, english_val_avg), dim=1)
german_val = torch.cat((german_val_m, german_val_avg), dim=1)

# Shape (1000, 768 x 2)
english_test = torch.cat((english_test_m, english_test_avg), dim=1)
german_test = torch.cat((german_test_m, german_test_avg), dim=1)


# Create Feature Vectors
# (en, ge, |en-ge|, en*ge)

# ** TRAIN **
en_ge_cat = torch.cat((english,german), dim=1)
en_ge_product = english * german
en_ge_abs_dif = (english - german).abs()

X_train = torch.cat((en_ge_cat, en_ge_product, en_ge_abs_dif), dim=1)

# ** VALIDATION **
en_ge_cat_val = torch.cat((english_val,german_val), dim=1)
en_ge_product_val = english_val * german_val
en_ge_abs_dif_val = (english_val - german_val).abs()

X_val = torch.cat((en_ge_cat_val, en_ge_product_val, en_ge_abs_dif_val), dim=1)

# ** TEST **

en_ge_cat_test = torch.cat((english_test,german_test), dim=1)
en_ge_product_test = english_test * german_test
en_ge_abs_dif_test = (english_test - german_test).abs()

X_test = torch.cat((en_ge_cat_test, en_ge_product_test, en_ge_abs_dif_test), dim=1)

def main():
    
    model = models.MLP_Regressor(activation='relu', regularization = 0.005, 
                    batch_size=128, hidden_layer_sizes=(4096, 2048, 1024,512, 256, 128), 
            learning_rate='adaptive',learning_rate_init=0.001, max_iter=25, n_iter_no_change=10,
             optimizer='adam', early_stopping=True, tol=0.0001, validation_fraction=0.15)
    
    model.fit(X_train, y_train)
    
    _, pearson, rmse = model.predict(X_val, y_val)
    
    print('Pearson: {}'.format(pearson))
    print('RMSE: {}'.format(rmse))
    
    
    
    

