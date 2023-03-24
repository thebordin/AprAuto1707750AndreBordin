import matplotlib.pyplot as plt #Importa a biblioteca matplotlib associada ao comando plt com a função de desenhar gráficos
import pickle as p1 #importa a biblioteca pickle associada ao comando p1 com a função de serializar o código.
import numpy as np #importa a biblioteca NumPy associada ao comando np com a função de suporte a grandes Datasets multi-dimensionais e matrizes assim como uma grande coleção de funções matemáticas de alto nível.
import pandas as pd #importa a biblioteca pandas associada ao comando pd com a função de facilitar a manipulação de dados a partir de diversos tipos de entrada.
from sklearn import datasets, linear_model #importa a biblioteca sklearn com as funções 1- Armazenar conjunto de datasets que podem ser importados. 2- de implementar varios modelos de calculos lineares
from sklearn.metrics import mean_squared_error, r2_score #importa a biblioteca como ferramentas de coeficiente de corelação
import random

contador=0 # Criar um contador ZERADO
while True:
    contador += 1                               #Contagem de ciclos até encontrar uma previsão correta.
    linha = random.randint(0,3824)              #Escolha de uma linha aleatória entre 0 e 3824.
    dataPre = pd.read_csv('dataset/optdigits.tra', sep=',', header=None)   #Abre a database optdigits.tra para pegar a linha escolhida como teste.
    data=dataPre.iloc[linha:linha+1,0:64]       #Seleciona os dados da linha para fazer a previsão
    charReal= dataPre.iloc[linha:linha+1,64]    #Seleciona a coluna da linha específica onde está o resultado real
    charReal= int(charReal.iloc[0])             #Converte a matriz do resultado real em uma lista

    loaded_model = p1.load(open('dataset/character_predictor', 'rb'))       #Abre o predictor criado no treino
    charPrevisto = int(loaded_model.predict(data))                          #Roda o predictor nos dados selecionados
    print('\nLinha: ', linha+1)
    print('Caractere Real: ',charReal,'Caractere Previsto: ', int(charPrevisto))
    if charReal == charPrevisto:
        print('Previsão CORRETA <==============================================================================')
        if charReal == charPrevisto : break     #Encerra o ciclo caso ache uma comparação exata.
    else: print('Previsão INCORRETA')
print('Foram ', contador, 'tentativas até um match.')