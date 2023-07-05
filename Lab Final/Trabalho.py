import os
from split_image import split_image
from PIL import Image


######Setup######
images_raw_loc = './input/'
images_split_loc = './output/splited/'
image_lined_loc = './output/lined/'
image_frame_loc = './output/'
image_frame_file = './output/image_frame.jpg'
p2n_loc = './output/output_text/'
dataset_file = './output/dataset/dataset.csv'
dataset_ss_file = './output/dataset/dataset_ss.csv'
trickbag_loc = './data/tickbag'
features_train_loc = './data/features_train.csv'
features_validation_loc = './data/features_validation.csv'
predictor_loc = './MLPC/MLPCpredictor.sav'
px = 25 # Pixels
proporcao_treino = 0.3
#################


#################### Read All Files
def RAF(input, output, order):
    dir_list = os.listdir(input)
    count=0
    dataset=""
    for name in dir_list:
        count =count+1
        order_out=order +' '+input+' '+name+' '+output
        os.system(order_out)
    print("Files and directories in '", output, "' :")
    print(os.listdir(output))

#################### Busca as subpastas, redimensiona, cria as linhas e monta o MNist
def IMAGEFRAME(images_raw, images_lined, images_frame):
    print('>Criando o MNinst:')
    for subfolder in os.listdir(images_raw):
        subfolder_path = os.path.join(images_raw, subfolder)
        print(f'>Redimensionando e criando uma linha de imagens de {subfolder_path}')
        if os.path.isdir(subfolder_path):
            images = []
            for filename in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, filename)
                image = Image.open(image_path)
                image_resized = image.resize((px, px))
                images.append(image_resized)
            merged_image = Image.new('RGB', (len(images)*px, px))
            x_offset = 0
            for image in images:
                merged_image.paste(image, (x_offset, 0))
                x_offset += px
            output_path = os.path.join(images_lined, subfolder + '.jpg')
            merged_image.save(output_path)
    print('>Criando o MNinst das imagens em linha:')
    frame = []
    for image_line in os.listdir(images_lined):
        image_path = os.path.join(images_lined, image_line)
        image = Image.open(image_path)
        frame.append(image)
    image_frame = Image.new('RGB', (10*px, len(frame)*px))
    y_offset = 0
    for lines in frame:
        image_frame.paste(lines, (0, y_offset))
        y_offset += px
    output_path = os.path.join(images_frame, 'image_frame.jpg')
    image_frame.save(output_path)
    print(f'>MNinst salvo em {output_path}\n')
    image_frame.show()

#IMAGEFRAME(images_raw_loc, image_lined_loc, image_frame_loc)

#################### Split
print('>Separando a imagem:')
#split_image(image_frame_file, 10, 10, False, False, False, output_dir=images_split_loc)
print('>Imagens separadas com sucesso.\n')

#################### P2N
print('>Convertendo imagens em CSV:')
order_p2n = 'python conversion_picture_to_numbers.py'
#RAF(images_split_loc,p2n_loc,order_p2n)
print('>Arquivos de imagem convertidos em csv com sucesso.\n')

#################### Dataset
print('>Criando Dataset:')
order_dataset = 'python dataset.py'
order_out=order_dataset +' '+p2n_loc+' '+dataset_file
os.system(order_out)
print('>Dataset criado com sucesso.\n')

#################### Verificando Missing Values
print('>Verificando dataset:')
order_dataset = 'python atv1.py'
order_out=order_dataset +' '+dataset_file
os.system(order_out)
print()

#################### Standard Scaler
print('>Padronizando entradas:')
order_ss = 'python atv2.py'
order_out=order_ss +' '+dataset_file+' '+dataset_ss_file+' '+trickbag_loc
os.system(order_out)
print('>Padronizacao de entradas concluido.\n')

#################### Divisão do dataset entre treino e teste
print('>Separação entre treino e teste:')
order_ss = 'python atv3.py'
order_out=order_ss +' '+dataset_ss_file+' '+trickbag_loc+' '+features_train_loc+' '+features_validation_loc+' '+str(proporcao_treino)
os.system(order_out)
print('>Separação entre treino e teste concluido.\n')

#################### MLPC treino:
print('>Treinando MLPC:')
order_ss = 'python atv4.py'
order_out=order_ss +' '+features_train_loc+' '+trickbag_loc+' '+predictor_loc
os.system(order_out)
print('>Treino Concluído.\n')

#################### Predictor:
print('>Rodando o predictor MLPC:')
order_ss = 'python atv5.py'
order_out=order_ss +' '+features_validation_loc+' '+predictor_loc
os.system(order_out)
print('>Treino Concluído.\n')




