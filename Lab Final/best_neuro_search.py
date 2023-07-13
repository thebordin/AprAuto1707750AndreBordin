import os
import random
import joblib
import pandas as pd
import numpy as np
from split_image import split_image
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier

######Setup######
url_images_raw = './input/'
url_images_lined = './output/lined/'
url_image_frame = './data/image_frame.jpg'
url_img_splited = './output/splited/'
url_p2n = './output/output_text/'
url_dataset = './data/dataset/dataset.csv'
url_dataset_ss = './data/dataset/dataset_ss.csv'
url_features_train = './data/features_train.csv'
url_features_validation = './data/features_validation.csv'
url_trickbag = './MLPC/MLPCpredictor.sav'
proporcao_treino_teste = 0.2
px = 25 # Pixels
#na,nb,nc = 75,115,75 #NEURONIOS
random_state = 42
max_iter = 100
min_neuron = 25
max_neuron = 200
#################
referencia = []
global count
# Conjunto de funções que vai desde a busca das imagens até o dataset dividido entre treino e teste
def Tratamento_Dados():
    # Busca as subpastas, redimensiona, cria as linhas e monta o MNist
    def MNist(url_images_raw, url_images_lined, url_image_frame):
        print('>Criando o MNinst:')
        count = 0
        for subfolder in os.listdir(url_images_raw):
            referencia.append(f'n: {count} = {subfolder}')
            count += 1
            subfolder_path = os.path.join(url_images_raw, subfolder)
            print(f'>Redimensionando e criando uma linha de imagens de {subfolder_path}')
            if os.path.isdir(subfolder_path):
                images = []
                for filename in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, filename)
                    image = Image.open(image_path)
                    image_resized = image.resize((px, px))
                    images.append(image_resized)
                merged_image = Image.new('RGB', (len(images) * px, px))
                x_offset = 0
                for image in images:
                    merged_image.paste(image, (x_offset, 0))
                    x_offset += px
                output_path = os.path.join(url_images_lined, subfolder + '.jpg')
                merged_image.save(output_path)
        print('>Criando o MNinst das imagens em linha:')
        frame = []
        for image_line in os.listdir(url_images_lined):
            image_path = os.path.join(url_images_lined, image_line)
            image = Image.open(image_path)
            frame.append(image)
        image_frame = Image.new('RGB', (10 * px, len(frame) * px))
        y_offset = 0
        for lines in frame:
            image_frame.paste(lines, (0, y_offset))
            y_offset += px
        image_frame.save(url_image_frame)
        print(f'>MNinst salvo em {url_image_frame}\n')
        image_frame.show()

    # Picture 2 Numbers
    def P2N(url_img_splited, url_p2n):
        def Convert(ulr_img_splited, name, url_p2n):
            img = Image.open(ulr_img_splited + name)
            arr = np.asarray(img)
            print(f'Convertendo {name}: Array Shape: {arr.shape}')
            lst = []
            for row in arr:
                tmp = []
                for col in row:
                    tmp.append(str(col))
                lst.append(tmp)
            with open(url_p2n + name + '.csv', 'w') as f:
                for row in lst:
                    f.write(','.join(row) + '\n')

        dir_list = os.listdir(url_img_splited)
        for name in dir_list:
            Convert(url_img_splited, name, url_p2n)
        print("Files and directories in '", url_p2n, "' :")
        print(os.listdir(url_p2n))

    # Dataset adicionando Labels
    def Dataset(url_p2n, url_dataset):
        referencia = ''

        def build(contents):
            nonlocal dataset
            w_array = contents.split(";")
            for w in w_array:
                w.replace('\n', ' ')
                ws = w.split(',')
                for z in ws:
                    z0 = z.replace('[  ', '')
                    z1 = z0.replace('[ ', '')
                    z2 = z1.replace('[', '')
                    z3 = z2.replace('   ', ' ')
                    z4 = z3.replace('  ', ' ')
                    zf = z4.replace(']', '')
                    u = zf.split(' ')
                    lst = []
                    for j in u:
                        cheat = ''.join(filter(str.isnumeric, j))
                        lst.append(cheat)
                    avg = (int(lst[0]) + int(lst[1]) + int(lst[2])) / 3
                    dataset += str(f'{float(avg)}') + ';'

        ds = open(url_dataset, "w")
        dataset = ''
        dir_list = os.listdir(url_p2n)
        print(dir_list)
        for each in dir_list:
            dataset = ''
            csv = url_p2n + each
            print(f'{each} Sendo adicionado ao DataSet')
            with open(csv) as f:
                content = f.readlines()
                for contents in content:
                    build(contents)
            name = each.split('_')
            number_raw = name[-1].split('.')
            number = int(number_raw[0])
            label = int(number / 10)
            dataset = dataset + str(label) + '\n'
            ds.write(dataset)
        ds.close()

    # Checando missing values
    def MVCheck(url_dataset):
        dataset = pd.read_csv(url_dataset, header=None, sep=';')
        #### FIM PARAMETROS ####
        cnt = 0
        for x in dataset.isna().any():
            if x == True:
                cnt += 1
        if cnt == 0:
            print('Dataset sem Missing Values')
        else:
            print(f'!!!!! DATASET COM {cnt} M/V !!!!!')

    # Aplicando Standart Scaler
    def SS(url_dataset, url_dataset_ss, url_trickbag):
        dataset = pd.read_csv(url_dataset, header=None, sep=';')
        predictor = Pipeline([('StandardScaler', StandardScaler())])
        predictor.steps.append(['referencia', referencia])
        predictor.steps.append(['px', px])
        dataset.iloc[:, :-1] = pd.DataFrame(predictor['StandardScaler'].fit_transform(dataset.iloc[:, 0:-1]))
        try:
            dataset.to_csv(url_dataset_ss, index=False)
        except:
            print('Houve um erro em importar o Dataset')

        try:
            predictor_file = open(url_trickbag, 'wb')
            joblib.dump(predictor, predictor_file)
            print('===== Modelo SS e Dataset Gravados =====')
        except:
            print('Houve um erro em importar o Modelo StandardScaler')

    # Divisão do dataset entre treino e teste
    def DSSplit(url_dataset_ss, url_features_train, url_features_validation, proporcao_treino_teste):
        rdm_state = random_state
        dataset = pd.read_csv(url_dataset_ss)
        features, features_validation = train_test_split(dataset,
                                                         test_size=proporcao_treino_teste,
                                                         random_state=rdm_state)

        try:
            features.to_csv(url_features_train, index=False)
            features_validation.to_csv(url_features_validation, index=False)
            print('Datasets separados e gravados com sucesso')
        except:
            print('Houve um problema na gravação dos datasets')

    print('>Buscando as imagens, tratando e montando o Frame:')
    MNist(url_images_raw, url_images_lined, url_image_frame)
    print('>Image Frame Concluido.\n')

    # Split
    print('>Dividindo o Frame:')
    split_image(url_image_frame, 10, 10, False, False, False, output_dir=url_img_splited)
    print('>Divisão Concluida.\n')

    print('>Convertendo imagem em números:')
    P2N(url_img_splited, url_p2n)
    print('>Conversão Concluida.\n')

    print('>Criando o Dataset e adicionando os Labels:')
    Dataset(url_p2n, url_dataset)
    print('>Criação do Dataset Concluida.\n')

    print('>Checando missing values:')
    MVCheck(url_dataset)
    print('>Checagem Concluida.\n')

    print('>Padronizando os dados de entrada:')
    SS(url_dataset, url_dataset_ss, url_trickbag)
    print('>Padronização Concluida.\n')

    print('>Dividindo o dataset entre treino e validação:')
    DSSplit(url_dataset_ss, url_features_train, url_features_validation, proporcao_treino_teste)
    print('>Divisão Concluida.\n')

# Função de treino do MLPC
def TreinoMLPC(url_features_train, url_trickbag,na, nb, nc):
    rdm_state = random_state
    predictor = joblib.load(open(url_trickbag, 'rb'))
    dataset = pd.read_csv(url_features_train)
    features = dataset.iloc[:, :-1]
    labels = dataset.iloc[:,-1]
    predictor.steps.append(['MLPC', MLPClassifier(hidden_layer_sizes=(na,nb,nc),
                                                 max_iter=max_iter,
                                                 activation= 'relu',
                                                 solver='adam',
                                                 random_state=rdm_state)])

    predictor['MLPC'].fit(features,labels)
    try:
        predictor_file = open(url_trickbag, 'wb')
        joblib.dump(predictor,predictor_file)
        print('Predictor gravado com sucesso')
        print(f'na: {na}, nb: {nb}, nc: {nc}')
    except: print('Houve um erro ao gravar o predictor')

def Avaliator():
    predictor = joblib.load(open(url_trickbag, 'rb'))
    dataset = pd.read_csv(url_features_validation)
    x_ss = dataset.iloc[:, :-1]
    y_validation = dataset.iloc[:, -1]

    y_predicted = predictor['MLPC'].predict(x_ss)
    y_predicted = pd.DataFrame(y_predicted)
    score = predictor['MLPC'].score(x_ss, y_validation)
    cm = confusion_matrix(y_predicted, y_validation)
    print('Traço = %.2f || Matriz confusão = %.2f || Score = %.2f || Precisao do MLPC = %.2f' % (
    cm.trace(), cm.sum(), score, accuracy_score(y_validation, y_predicted)))
    return score

# Função para tratar os dados
Tratamento_Dados()

def Avaliacao(na,nb,nc):
    TreinoMLPC(url_features_train, url_trickbag, na, nb, nc)
    return Avaliator()



# Função de criação de indivíduo aleatório
def criar_individuo():
    return (random.randint(min_neuron, max_neuron), random.randint(min_neuron, max_neuron), random.randint(min_neuron, max_neuron))

# Função de criação de população inicial
def criar_populacao(tamanho_populacao):
    return [criar_individuo() for _ in range(tamanho_populacao)]

# Função de cruzamento de dois indivíduos
def cruzar(individuo1, individuo2):
    return (
        random.choice([individuo1[0], individuo2[0]]),
        random.choice([individuo1[1], individuo2[1]]),
        random.choice([individuo1[2], individuo2[2]])
    )

# Função de mutação de um indivíduo
def mutar(individuo):
    return (
        max(1, min(100, individuo[0] + random.randint(-10, 10))),
        max(1, min(100, individuo[1] + random.randint(-10, 10))),
        max(1, min(100, individuo[2] + random.randint(-10, 10)))
    )

# Função de seleção de indivíduos para reprodução
def selecionar(populacao, tamanho_elite):
    populacao_ordenada = sorted(populacao, key=lambda individuo: Avaliacao(*individuo), reverse=True)
    return populacao_ordenada[:tamanho_elite]

# Parâmetros do algoritmo genético
tamanho_populacao = 70
tamanho_elite = 7
num_geracoes = 2

# Criar população inicial
populacao = criar_populacao(tamanho_populacao)

# Iterar pelas gerações
for geracao in range(num_geracoes):
    # Selecionar indivíduos para reprodução
    elite = selecionar(populacao, tamanho_elite)

    # Realizar cruzamento e mutação para gerar nova população
    nova_populacao = elite[:]
    while len(nova_populacao) < tamanho_populacao:
        print(f'Geração: {geracao}')
        individuo1 = random.choice(elite)
        individuo2 = random.choice(elite)
        novo_individuo = cruzar(individuo1, individuo2)
        novo_individuo = mutar(novo_individuo)
        nova_populacao.append(novo_individuo)

    # Atualizar a população
    populacao = nova_populacao

# Avaliar a melhor solução encontrada
melhor_individuo = max(populacao, key=lambda individuo: Avaliacao(*individuo))
melhor_na, melhor_nb, melhor_nc = melhor_individuo

# Imprimir a melhor configuração encontrada
print("Melhor configuração:")
print("na:", melhor_na)
print("nb:", melhor_nb)
print("nc:", melhor_nc)
print("Melhor score:", Avaliacao(melhor_na, melhor_nb, melhor_nc))