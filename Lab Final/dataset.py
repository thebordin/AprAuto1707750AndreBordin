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
    w_array = contents.split(";")
    #print("conteudo_aux2= ",conteudo_aux2)
    for w in w_array:
        w0 = w.replace('  ', ' ')
        w1 = w0.replace(']','')
        w2 = w1.replace('[', '')
        w3 = w2.replace('\n', '')
        w5 = w3.split(",")

        for z in w5:
            u = z.split(' ')
            lst = []
            for j in u:
                if j != ' ':
                    dataset = dataset+str(j)
            '''avg = (lst[0] + lst[1] + lst[2])
            print(avg)
            dataset=dataset+str(avg)'''
            if z != w5[-1]:
                dataset=dataset+';'
            

for each in dir_list:
    csv = input + each
    with open(csv) as f:
        content = f.readlines()
        for contents in content:
            build(contents)
    dataset=dataset+"\n"
ds = open(output, "w")
ds.write(dataset)
ds.close()
