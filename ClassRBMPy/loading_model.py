import os
import pdb
import rbm
import numpy

def matrix_load(path):
    fobj = open(path, 'r')
    line = fobj.readline()
    line = line.strip()
    row = int(line.split("\t")[0])
    col = int(line.split("\t")[1])

    matrix = []
    for i in range(0, row):
        line = fobj.readline()
        line = line.strip()
        tmp = line.split('\t')
        if len(tmp) != col:
            print "Invalid File, file col:%d  define col:%d"%(len(tmp), col)
            exit(-1)
        for j in range(0, col):
            tmp[j] = eval(tmp[j])
        matrix.append(tmp)
    return numpy.array(matrix)

def matrix_save(path, matrix):
    wobj = open(path, 'w')
    wobj.write(str(numpy.size(matrix,0))+"\t"+str(numpy.size(matrix,1))+"\n")
    for i in range(0, numpy.size(matrix,0)):
        for j in range(0, numpy.size(matrix,1)):
            wobj.write(str(matrix[i][j])+"\t")
        wobj.write('\n')
    wobj.close()

def loading_model(path):
    model = rbm.Model()
    model.W = matrix_load(path+'/W.txt')
    model.b  = matrix_load(path+'/b.txt')
    model.c  = matrix_load(path+'/c.txt')
    model.Wc = matrix_load(path+'/Wc.txt')
    model.cc = matrix_load(path+'/cc.txt')
    model.labels = matrix_load(path+'/labels.txt')
    return model

def save_model(path, model):
    if not os.path.exists(path):
        os.mkdir(path)
    matrix_save(path+'/W.txt', model.W)
    matrix_save(path+'/b.txt', model.b)
    matrix_save(path+'/c.txt', model.c)
    matrix_save(path+'/Wc.txt', model.Wc)
    matrix_save(path+'/cc.txt', model.cc)
    matrix_save(path+'/labels.txt', model.labels)





if __name__ == '__main__':
    model = loading_model('./model/test_model/')


