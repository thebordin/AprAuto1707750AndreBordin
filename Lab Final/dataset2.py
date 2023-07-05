import os
import sys
path = "./output_text/"
dir_list = os.listdir(path)
count=0
dataset=""
count==-1
content=""
if len(sys.argv)>1:
    u=sys.argv[1]
else:
    u="s1707272_0.jpg.csv"

print(dir_list[0])
x=dir_list[0]
u="./output_text/"+x
content_final=""

def build(contents):
    global dataset;
    conteudo_aux0 = contents.replace("\n", "")
    conteudo_aux1 = conteudo_aux0.replace(",", ";")
    conteudo_aux2_1 = conteudo_aux1.replace(" ", ",")
    conteudo_aux2_2 = conteudo_aux2_1.replace(",,", ",")
    conteudo_aux2_3 = conteudo_aux2_2.replace("[,", "[")
    conteudo_aux2 = conteudo_aux2_3.replace(",]", "]")
    w_array = conteudo_aux2.split(";")
    print("******************")
    lw_array=len(w_array)
    print("conteudo_aux2= ",conteudo_aux2)
    for w in w_array:
        w1=w.replace("[","")
        w2 = w1.replace("]", "")
        w3=w2.split(",")
        print(w3[0],",",w3[1],",",w3[2],",",str(int((int(w3[0])+int(w3[1])+int(w3[2]))/3)))
        dataset=dataset+";"+str(int((int(w3[0])+int(w3[1])+int(w3[2]))/3))
count=0 #
for u in dir_list:
    v="./output_text/" + u
    with open(v) as f:
        content = f.readlines()
        for contents in content:
            build(contents)
    dataset=dataset+"\n"
    count=count+1 #
    if count>=50: #
        break #

#airbnb_data = pd.read_csv("./"+dir)
"""
for x in dir_list:
    count=count+1
    print("count= ",count,x)
    if count==1:
        u="./output_text/"+x
        with open(u) as f:
            print()
            content = f.readlines()

print(len(content),content)
content_length=len(content)
conteudo=content[0].replace("\n","")

dataset=""


def build_dataset(n):
    global dataset;
    conteudo_aux0 = content[n].replace("\n", "")
    conteudo_aux1 = conteudo_aux0.replace(",", ";")
    conteudo_aux2_1 = conteudo_aux1.replace(" ", ",")
    conteudo_aux2_2 = conteudo_aux2_1.replace(",,", ",")
    conteudo_aux2_3 = conteudo_aux2_2.replace("[,", "[")
    conteudo_aux2 = conteudo_aux2_3.replace(",]", "]")
    w_array = conteudo_aux2.split(";")
    print("******************")
    lw_array=len(w_array)
    print("conteudo_aux2= ",conteudo_aux2)
    for w in w_array:
        w1=w.replace("[","")
        w2 = w1.replace("]", "")
        w3=w2.split(",")
        print(w3[0],",",w3[1],",",w3[2],",",str(int((int(w3[0])+int(w3[1])+int(w3[2]))/3)))
        dataset=dataset+";"+str(int((int(w3[0])+int(w3[1])+int(w3[2]))/3))

build_dataset(0)
build_dataset(1)
build_dataset(2)
print(content_length)
#content_length=content_length-20
for u in range(1,content_length):
    print("u=",u)
    print("****************")
    print(content[u])
    build_dataset(u)
"""

h = open("./dataset_texto/dataset_final"+x, "a")
h.write(dataset)
h.close()
print("Files and directories in '", path, "' :")

print(dir_list)
