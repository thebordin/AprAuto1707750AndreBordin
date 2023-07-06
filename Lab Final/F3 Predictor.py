import joblib
import pandas as pd
import numpy as np
from PIL import Image

######Setup######
url_sample = './sample/sample.jpg'
url_p2n_sample = './sample/data/p2n_csv.csv'
url_dataset_sample = './sample/data/dataset_sample.csv'
url_dataset_sample_ss = './sample/data/dataset_sample_ss.csv'
url_trickbag = './MLPC/MLPCpredictor.sav'
predictor = joblib.load(open(url_trickbag, 'rb'))
referencia = predictor['referencia']
#################

def P2N(url_sample, url_p2n_sample):
    img = Image.open(url_sample).resize((25,25))
    arr = np.asarray(img)
    print(f'Convertendo {url_sample}: Array Shape: {arr.shape}')
    lst = []
    for row in arr:
        tmp = []
        for col in row:
            tmp.append(str(col))
        lst.append(tmp)
    with open(url_p2n_sample, 'w') as f:
        for row in lst:
            f.write(','.join(row) + '\n')

def Dataset(url_sample,url_dataset_sample):
    dataset = ''
    def build(contents):
        nonlocal dataset
        w_array = contents.split(";")
        for w in w_array:
            w.replace('\n',' ')
            ws = w.split(',')

            for z in ws:
                z0 = z.replace('[  ','')
                z1 = z0.replace('[ ','')
                z2 = z1.replace('[','')
                z3 = z2.replace('   ',' ')
                z4 = z3.replace('  ',' ')
                zf = z4.replace(']','')
                u = zf.split(' ')
                lst = []
                for j in u:
                    cheat = ''.join(filter(str.isnumeric, j))
                    lst.append(cheat)
                avg = (int(lst[0]) + int(lst[1]) + int(lst[2]))/3
                dataset += str(f'{float(avg)}')
                dataset=dataset+';'
    print(f'{url_p2n_sample} Sendo adicionado ao DataSet')
    with open(url_p2n_sample) as f:
        content = f.readlines()
        for contents in content:
            build(contents)
    ds = open(url_dataset_sample, "w")
    ds.write(dataset[:-1])
    ds.close()

def MVCheck(url_dataset_sample):
    dataset = pd.read_csv(url_dataset_sample, header=None, sep=';')
    #### FIM PARAMETROS ####
    cnt = 0
    for x in dataset.isna().any():
        if x == True:
            cnt += 1
    if cnt == 0:
        print('Dataset sem Missing Values')
    else:
        print (f'!!!!! DATASET COM {cnt} M/V !!!!!')

def SS(url_dataset_sample,url_dataset_sample_ss):
    dataset = pd.read_csv(url_dataset_sample, header=None, sep=';')
    dataset = pd.DataFrame(predictor['StandardScaler'].transform(dataset))
    try:
        dataset.to_csv(url_dataset_sample_ss, index = False)
    except: print('Houve um erro em importar o Dataset')

def Predict(url_dataset_sample_ss):
    dataset = pd.read_csv(url_dataset_sample_ss)
    x_ss = dataset.iloc[:, :]
    y_predicted = predictor['MLPC'].predict(x_ss)
    print(f'Previsão: {y_predicted}\nReferencia: \n{referencia}')

P2N(url_sample, url_p2n_sample)
Dataset(url_p2n_sample, url_dataset_sample)
MVCheck(url_dataset_sample)
SS(url_dataset_sample,url_dataset_sample_ss)
print('##########################')
print(f'Previsão para {url_sample} por {url_trickbag}:')
Predict(url_dataset_sample_ss)