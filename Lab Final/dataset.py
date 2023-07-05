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
        w0 = w.replace('   ', '  ')
        w1 = w0.replace('  ',' ')
        w2 = w1.replace('[ ', '')
        w2 = w2.replace('[', '')
        w3 = w2.replace(']', '')
        w5 = w3.split(",")

        for z in w5:
            u = z.split(' ')
            lst = []
            for j in u:
                lst.append(j)
            avg = (int(lst[0]) + int(lst[1]) + int(lst[2]))/3
            dataset=dataset+str(int(avg))
            if z != w5[-1]:
                dataset=dataset+';'

for each in dir_list:
    csv = input + each
    print(f'{each} Sendo adicionado ao DataSet')
    with open(csv) as f:
        content = f.readlines()
        for contents in content:
            build(contents)
    name = each.split('_')
    number_raw = name[-1].split('.')
    number = int(number_raw[0])
    label = int(number/10)
    dataset=dataset+';'+str(label)+'\n'
ds = open(output, "w")
ds.write(dataset)
ds.close()
