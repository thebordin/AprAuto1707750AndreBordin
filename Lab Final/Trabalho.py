import os
from split_image import split_image
from PIL import Image


######Setup######
images_raw_loc = './input/'
images_split_loc = './output/splited/'
image_lined_loc = './output/lined/'
image_frame_loc = './output/'
px = 250 # Pixels
#################


#################### Read All Files
def RAF(input, output, order):
    dir_list = os.listdir(input)
    count=0
    dataset=""
    for name in dir_list:
        count =count+1
        print(name,type(name))
        order_out=order +' '+input+' '+name+' '+output
        os.system(order_out)
    print("Files and directories in '", output, "' :")
    print(dir_list)

'''images_raw_list = os.listdir(images_raw_loc)
count=0
dataset=""
for x in images_raw_list:
    count =count+1
    print(x,type(x))
    order="python conversion_picture_to_numbers.py "+x
    os.system(order)

print("Files and directories in '", images_raw_loc, "' :")
print(images_raw_loc)'''

'''#################### Resize
order_resize = 'python resize_image.py'
RAF(images_raw_loc, images_resized_loc, order_resize)'''

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
    print(f'>MNinst salvo em {output_path}')
    image_frame.show()
IMAGEFRAME(images_raw_loc, image_lined_loc, image_frame_loc)
print('>Separando a imagem:')


'''
file=sys.argv[1]+sys.argv[2]
split_image(file,10, 10, False, False,output_dir=sys.argv[3])
#split_image(image_path, rows, cols, should_square, should_cleanup, [output_dir])
# e.g. split_image("bridge.jpg", 2, 2, True, False)'''