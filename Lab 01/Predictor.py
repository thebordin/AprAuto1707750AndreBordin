import pickle as p1 #importa a biblioteca pickle associada ao comando p1 com a função de serializar o código.
import pandas as pd #importa a biblioteca pandas associada ao comando pd com a função de facilitar a manipulação de dados a partir de diversos tipos de entrada.
import random

dataPre = pd.read_csv('dataset/optdigits.tra', sep=',',header=None)  # Abre a database optdigits.tra para pegar a linha escolhida como teste.
loaded_model = p1.load(open('dataset/character_predictor', 'rb'))  # Abre o predictor criado no treino
contador=0 # Criar um contador ZERADO

while True:
    contador += 1                               #Contagem de ciclos até encontrar uma previsão correta.
    linha = random.randint(0,3824)              #Escolha de uma linha aleatória entre 0 e 3824.
    data=dataPre.iloc[linha:linha+1,0:64]       #Seleciona os dados da linha para fazer a previsão
    charReal= dataPre.iloc[linha:linha+1,64]    #Seleciona a coluna da linha específica onde está o resultado real
    charReal= int(charReal.iloc[0])             #Converte a matriz do resultado real em uma lista
    charPrevisto = int(loaded_model.predict(data))                          #Roda o predictor nos dados selecionados
    print('\nLinha: ', linha+1)
    print('Caractere Real: ',charReal,'Caractere Previsto: ', int(charPrevisto))
    if charReal == charPrevisto:
        print('Previsão CORRETA <==============================================================================')
        if charReal == charPrevisto : break     #Encerra o ciclo caso ache uma comparação exata.
    else: print('Previsão INCORRETA')
print('Foram ', contador, 'tentativas até um match.')