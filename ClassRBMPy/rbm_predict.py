import numpy
import pdb
import rbm
import math


#yhat = rbm_predict.predict(model, testdata)

def get_ith_col(x, i):
    t = numpy.transpose(x)
    return numpy.transpose([t[i]])

def predict(model, testdata):
    """
    input:
    @model          is the model from rbm, and it is defined by class Model.
    @testdata       binary, or in [0,1] intyerpreted as probailities.
    output:
    @prediction     the discrete labels for every class
    """

    numclasses = numpy.size(model.Wc, 0)
    numcases = numpy.size(testdata, 0)

    F = numpy.zeros((numclasses, numcases))
    X = numpy.zeros((numcases, numclasses))
    for i in range(0, numclasses):
        X = numpy.zeros((numcases, numclasses))
        for j in range(0, numcases):
            X[j][i] = 1
        line = numpy.transpose(numpy.tile(model.cc[0][i],[numcases,1])*get_ith_col(X,i)+ numpy.transpose([numpy.sum(numpy.log(numpy.exp(numpy.dot(testdata, model.W)+numpy.dot(X, model.Wc)+numpy.tile(model.b,[numcases,1])) +1),axis=1)]))
        F[i] = line

    F = numpy.transpose(F)
    p = []
    q = []

    #pdb.set_trace()
    for i in range(0, numcases):
        maxval = -10000000
        pos = -1
        for k in range(0, numclasses):
            if maxval < F[i][k]:
                maxval = F[i][k]
                pos = k
        p.append(maxval)
        q.append(pos)

    prediction = []
    for i in range(0, len(q) ):
        prediction.append(model.labels[q[i]][0])
    return prediction



