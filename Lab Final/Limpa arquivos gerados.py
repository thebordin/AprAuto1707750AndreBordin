import os

# Folders a serem limpas:
url_images_lined = './output/lined/'
url_img_splited = './output/splited/'
url_p2n = './output/output_text/'
# Ficheiros a serem apagados
url_image_frame = './data/image_frame.jpg'
url_dataset = './data/dataset/dataset.csv'
url_dataset_ss = './data/dataset/dataset_ss.csv'
url_features_train = './data/features_train.csv'
url_features_validation = './data/features_validation.csv'
url_trickbag = './MLPC/MLPCpredictor.sav'

pastas = [url_images_lined,url_img_splited,url_p2n]
ficheiros = [url_image_frame, url_dataset,url_dataset_ss,url_features_train,url_features_validation,url_trickbag]

for pasta in pastas:
    for nome_arquivo in os.listdir(pasta):
        caminho_arquivo = os.path.join(pasta, nome_arquivo)
        if os.path.isfile(caminho_arquivo):
            print(f'>Apagando {caminho_arquivo}.')
            os.remove(caminho_arquivo)

for ficheiro in ficheiros:
    if os.path.isfile(ficheiro):
        print(f'>Apagando {ficheiro}.')
        os.remove(ficheiro)