'Atividade 1'
#Indique a lógica e o procedimento de cada linha do código em baixo.

import matplotlib.pyplot as plt #Importa a biblioteca matplotlib associada ao comando plt com a função de desenhar gráficos
import pickle as p1 #importa a biblioteca pickle associada ao comando p1 com a função de serializar o código.
import numpy as np #importa a biblioteca NumPy associada ao comando np com a função de suporte a grandes Datasets multi-dimensionais e matrizes assim como uma grande coleção de funções matemáticas de alto nível.
import pandas as pd #importa a biblioteca pandas associada ao comando pd com a função de facilitar a manipulação de dados a partir de diversos tipos de entrada.
from sklearn import datasets, linear_model #importa a biblioteca sklearn com as funções 1- Armazenar conjunto de datasets que podem ser importados. 2- de implementar varios modelos de calculos lineares
from sklearn.metrics import mean_squared_error, r2_score #importa a biblioteca como ferramentas de coeficiente de corelação
data = pd.read_csv("../VinhoB/Qualidade_vinho_B/winequality-white.csv", sep=";") #Lê o arquivo em especifico usando o Pandas e atribui o separador de dados.

#Atividade 2
#Indique a lógica e o procedimento de cada linha do código em baixo.
train_data=data[:1000] #Seleciona as primeiras 1000 linhas do conjunto de dados e armazena essas linhas na variavel train_data.
data_X=train_data.iloc[:,0:11] #Utilizando o comando iloc, armazena em data_X as colunas de 0-10 de train_data.
data_Y=train_data.iloc[:,11:12] #Utilizando o comando iloc, armazena em data_Y a coluna 11 de train_data.
#print(train_data.columns) #Provavelmente uma ferramenta para verificação dos dados armazenados em train_data. Apresenta "#", ou seja, é um comando que está desativado.
print(data_X) #Mostra os dados em data_X
print(data_Y) #Mostra os dados em data_Y

#Atividade 3
#Indique a lógica e o procedimento de cada linha do código em baixo.
regr = linear_model.LinearRegression() #Cria um objeto com a função de calculo de regressão linar.
preditor_linear_model=regr.fit(data_X, data_Y) #Invoca data_X e data_Y para o objeto regr e dá o nome de preditor_linear_model.
preditor_Pickle = open('../white-wine_quality_predictor', 'wb') #Abre um arquivo no modo de gravação (w) binária (b) para armazenar o modelo treinado.
print("white-wine_quality_predictor") #Imprime no ecrã "white-wine_quality_predictor"
p1.dump(preditor_linear_model, preditor_Pickle) #Fecha o arquivo de gravação binário
'import.... + data' #Refere-se a totalidade dos imports da ATIVIDADE 1 sendo que agora é feito em outro ficheiro para testar o mecanismo treinado.
evaluation_data=data[1001:] #Seleciona as linhas a partir da posição 1001 do conjunto de dados e armazena essas linhas na variavel evaluation_data.
data_X=evaluation_data.iloc[:,0:11] #Utilizando o comando iloc, armazena em data_X as colunas de 0-10 de evaluation_data.
data_Y=evaluation_data.iloc[:,11:12] #Utilizando o comando iloc, armazena em data_X a coluna 11 de evaluation_data.
print(type(evaluation_data)) #Imprime o tipo de dados armazenado em evalution_data
print(type(data_X)) #Imprime o tipo de dados armazenado em data_X
loaded_model = p1.load(open('../white-wine_quality_predictor', 'rb')) #Carrega para a variavel loaded_model o modelo armazenado anteriormente.
print("Coefficients: \n", loaded_model.coef_) #Imprime os coeficientes do modelo preditor carregado.
y_pred=loaded_model.predict(data_X) #faz uma previsão de Y a partir de X utilizando o modelo preditor carregado.
z_pred=y_pred-data_Y #Calcula a diferença entre o valor previsto (Y_pred) e o valor real (data_Y) e armazena na variavel z_pred.

#Atividade 4
#Indique a lógica e o procedimento de cada linha do código em baixo.

right, wrong, total=0 #Define as 3 variaveis com o valor inicial "0".
for x in z_pred["quality"]: #Na coluna "quality" de z_pred faz um loop com as funções:
    z=int(x) #Armazena em z o valor de x convertendo para inteiro.
    total=total+1 #Incrementa a variavel total em 1.
    if z==0: #Se z for igual a zero (diferença entre data_Y e y_pred)
        right=right+1 #Incrementa a variavel right em 1.
    else: #Senão
        wrong=wrong+1 #Incrementa a variavel wrong em 1.
print("accuraccy1= ",right/total,"accuraccy2= ",wrong/total) #Imprime a frequencia relativa de right e wrong.

#Atividade 5
#Indique a lógica e o procedimento de cada linha do código em baixo.

'import....' #Novamente refere-se a importar as bibliotecas vistas na Atividade 1.
data_x=input("introduza valores do wine\n") #Pede ao usuario introduzir os valores do wine e armazena em data_x.
data=data_x.split(";") #Define ";" como separador e armazena os dados já separados em data.
print(data) #Imprime o conteudo de data.
fmap_data = map(float, data) #Atravez da função map() transforma todos os elementos de data em dados do tipo float, armazenando em fmap_data.
print(fmap_data) #Imprime o conteudo de fmap_data.
flist_data = list(fmap_data) #Converte fmap_data em uma lista de floats e armazena em flist_data
print(flist_data) #Imprime o conteudo de flist_data
data1 = pd.read_csv("../VinhoB/Qualidade_vinho_B/winequalitywhite.csv",sep=";") #Armazena os dados do arquivo utilizando ";" como separação em data1.
data2=data1.iloc[:0,:11] #Armazena em data 2 as colunas de 0-10 de data1.
data_preparation=pd.DataFrame([flist_data],columns=list(data2)) #Cria um DataFrame "data_preparation" a partir de flist_data com as colunas que constam em data2.
out=data2 #out recebe o valor de data2
for x in out: #Cria um loop com out ciclos para:
    print(x,data_preparation[x].values) #Imprimir: Numero do ciclo, Valor correspondente da linha em data_preparation
    loaded_model = p1.load(open('../white-wine_quality_predictor', 'rb')) #Armazena em loaded_model o modelo binário carregado.
    y_pred=loaded_model.predict(data_preparation)  #Usa o modelo carregado para prever os valores a partir de data_preparation armazenando os resultados em y_pred.
    print("wine quality",int(y_pred)) #Imprime a previsão calculada