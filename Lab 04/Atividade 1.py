import pandas as pd

#### PARAMETROS ####
url_dataset = 'data/heart.csv'
url_ds_clean = 'data/heart_clean_1.csv'
#### FIM PARAMETROS ####

dataset = pd.read_csv(url_dataset)
cnt = 0
for x in dataset.isna().any():
    if x == True:
        cnt += 1
if cnt == 0:
    try:
        dataset.to_csv(url_ds_clean, index=False)
        print('Dataset salvo sem Missing Values')
    except: print('Houve um erro na gravação')
else:
    print ('Dataset não salvo pois existem Missing Values')