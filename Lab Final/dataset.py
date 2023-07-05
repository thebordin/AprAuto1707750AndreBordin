import os
import sys

# 1. Setup
input = sys.argv[1]
output = sys.argv[2]

dir_list = os.listdir(input)
print(dir_list)
dataset=""
content=""

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
        w1 = w.replace("[","")
        w2 = w1.replace("]", "")
        w3 = w2.split(",")
        for i in range(3): ###################
            if w3[i]=='':
                w3[i]=0
        w4 = int( (int(w3[0]) + int(w3[1]) + int(w3[2]) ))
        #print(w3[0],",",w3[1],",",w3[2],",",str(int((int(w3[0])+int(w3[1])+int(w3[2]))/3)))
        dataset=dataset+str(int(w4/3))
        if w != w_array[-1]:
            dataset=dataset+';'
            


count=0
for each in dir_list:
    csv = input + dir_list[count]
    with open(csv) as f:
        content = f.readlines()
        for contents in content:
            build(contents)
    dataset=dataset+"\n"
    count=count+1
    '''if count>=50: #
        break #'''
ds = open(output, "w")
ds.write(dataset)
ds.close()
