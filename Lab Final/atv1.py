import pandas as pd
import sys

#### PARAMETROS ####
input = sys.argv[1]
dataset = pd.read_csv(input, header=None, sep=';')
#### FIM PARAMETROS ####
cnt = 0
for x in dataset.isna().any():
    if x == True:
        cnt += 1
if cnt == 0:
    print('Dataset sem Missing Values')
else:
    print ('!!!!! DATASET COM M/V !!!!!')
