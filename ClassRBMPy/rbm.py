import numpy
import loading_model
import math

import pdb

def logistic(x):
    x = numpy.array(x)
    return 1.0 / (1.0 + numpy.exp(-x))

def softmax_pmtk(eta):
    # softmax function
    # mu[i][c] = exp(eta[i][c]) / sum_c'  exp(eta[i][c'])
    tmp = numpy.exp(eta)
    D = len(eta)
    C = len(eta[0])
    gen = numpy.zeros((C, 1))+1
    denom = numpy.dot(tmp, gen)
    mu = tmp / numpy.tile(denom, [1,C])
    return mu

def softmax_sample(probmat):
    row = len(probmat)
    col = len(probmat[0])
    oneofn = numpy.zeros((row, col))
    #tmp_one = numpy.zeros((col, 1)) + 1
    #sum_probmat = numpy.dot(probmat, tmp_one)
    sum_probmat = numpy.sum(probmat, axis = 1)
    sum_probmat = numpy.transpose([sum_probmat])
    probmat = probmat / numpy.tile(sum_probmat, [1, col])
    for i in range(0, row):
        probs = probmat[i]
        sample = numpy.cumsum(probs)
        sample = sample > numpy.random.rand()
        # find the first true value elem in sample
        index = -1
        for j in range(0, len(sample)):
            if sample[j]:
                index = j
                break
        sample = numpy.zeros((1, len(probs)) )
        sample[0][index] = 1
        oneofn[i] = sample
    return oneofn

class Model:
    def __init__(self):
        self.W = None
        self.c = None
        self.b = None
        self.Wc = None
        self.cc = None
        self.ph = None
        self.nh = None
        self.phstates = None
        self.nhstates = None
        self.negdata = None
        self.negdatastates = None
        self.Winc = None
        self.binc = None
        self.Wcinc = None
        self.ccinc = None
        self.Wavg = None
        self.bavg = None
        self.cavg = None
        self.ccavg = None

class RBM:
    def __init__(self, con):
        self.in_params = con.get_params()
        self.out_params = {}
        self.out_params['W'] = None                     # the weight of the connections
        self.out_params['b'] = None                     # the biases of the hidden layer
        self.out_params['c'] = None                     # the biases of the visible layer
        self.out_params['Wc']= None                     # the weights on labels layer
        self.out_params['cc']= None                     # the biases on labels layer

        self.errors = None                              # errors in reconstruction at every epoch

    def encoding(self):
        """
        create classes layer code: 1-of-k encoding for each discrete label
        """
        row_x = numpy.size(self.in_params['X'], 0)
        col_x = numpy.size(self.in_params['X'], 1)

        targets = numpy.zeros((row_x, self.in_params['nclasses']))
        u = numpy.unique(self.in_params['y'])
        for j in range(0, len(self.in_params['y']) ):
            label = self.in_params['y'][j]
            for i in range(0, len(u)):
                if label == u[i]:
                    targets[j][i] = 1;
        return targets

    def initialize_rbm(self):
        numcases = numpy.size(self.in_params['X'], 0)
        numdims =  numpy.size(self.in_params['X'], 1)
        numclasses = self.in_params['nclasses']
        numhid = self.in_params['numhid']

        model = Model()
        #model.W = numpy.dot(1, numpy.random.randn(numdims, numhid))
        model.W = loading_model.matrix_load('test_data/W_init.txt')
        model.c = numpy.zeros((1, numdims))
        model.b = numpy.zeros((1, numhid))
        #model.Wc = numpy.dot(1, numpy.random.randn(numclasses, numhid))
        model.Wc = loading_model.matrix_load('test_data/Wc_init.txt')
        model.cc = numpy.zeros((1, numclasses))
        model.ph = numpy.zeros((numcases, numhid))
        model.nh = numpy.zeros((numcases, numhid))
        model.phstates = numpy.zeros((numcases, numhid))
        model.nhstates = numpy.zeros((numcases, numhid))

        model.negdata = numpy.zeros((numcases, numdims))
        model.negdatastates = numpy.zeros((numcases, numdims))
        model.Winc = numpy.zeros((numdims, numhid))
        model.binc = numpy.zeros((1, numhid))
        model.cinc = numpy.zeros((1, numdims))
        model.Wcinc = numpy.zeros((numclasses, numhid))
        model.ccinc = numpy.zeros((1, numclasses))
        model.Wavg = model.W
        model.bavg = model.b
        model.cavg = model.c
        model.Wcavg = model.Wc
        model.ccavg = model.cc
        self.errors = numpy.zeros((1, self.in_params['maxepoch']))
        return model



    def train(self):
        """
        train the RBM.
        """
        maxepoch = self.in_params['maxepoch']
        avglast = self.in_params['avglast']
        penalty = self.in_params['penalty']
        momentum = self.in_params['momentum']
        numhid = self.in_params['numhid']
        eta = self.in_params['eta']

        avgstart = maxepoch - avglast
        oldpenalty = penalty
        targets = self.encoding()
        # use min-batch, create batches
        row_X = numpy.size(self.in_params['X'], 0)
        col_X = numpy.size(self.in_params['X'], 1)
        numbatches = row_X / self.in_params['batchsize']
        row_X = row_X-row_X%self.in_params['batchsize']
        batchdata = []
        batchtargets = []
        for i in range(0, numbatches):
            batchdata.append([])
            batchtargets.append([])
        for j in range(0, numbatches):
            for i in range(0,row_X):
                if i%numbatches == j:
                    batchdata[j].append(list(self.in_params['X'][i]))
                    batchtargets[j].append(list(targets[i]))

        # start fitting RBM
        model = self.initialize_rbm()
        t = 1
        anneal = self.in_params['anneal']
        for epoch in range(0, maxepoch):
            print "epoch:%d"%epoch
            errsum = 0
            if anneal:
                penalty= oldpenalty - 0.9*epoch/maxepoch*oldpenalty;
            for batch in range(0, numbatches):
                #print "total batch number:%d current:%d"%(numbatches, batch)
                data = batchdata[batch]
                numcases = len(data)
                numdims = len(data[0])
                classes = batchtargets[batch]

                # positive phase, h_i given x_i and y_i
                model.ph  = logistic(numpy.dot(data, model.W)+numpy.dot(classes, model.Wc)+numpy.tile(model.b,[numcases,1]) )
                random_mat = numpy.random.rand(numcases, numhid)
                model.phstates = model.ph > random_mat
                model.nhstates = model.phstates

                # negtive phase, x_{i+1} given h_{i}
                model.negdata = logistic(numpy.dot(model.nhstates, numpy.transpose(model.W)) + numpy.tile(model.c, [numcases, 1]))
                random_mat = numpy.random.rand(numcases, numdims)
                model.negdatastates = model.negdata > random_mat

                # y_{i+1} given h_{i}
                # we use the softmax in order to keep a set of binary units whose states are mutually constrained so that exactly one of the K states has value 1 and the rest has 0.
                model.negclasses = softmax_pmtk(numpy.dot(model.nhstates, numpy.transpose(model.Wc)) + numpy.tile(model.cc, [numcases, 1]))
                model.negclassesstates = softmax_sample(model.negclasses)

                # go up one more time
                # h_{i+1} given x_{i+1] and y_{i+1}
                model.nh = logistic(numpy.dot(model.negdatastates, model.W)+ numpy.dot(model.negclassesstates, model.Wc)+ numpy.tile(model.b,[numcases,1]) )
                model.nhstates = model.nh > numpy.random.rand(numcases, numhid)


                # update weights and biases
                model.dW = (numpy.dot(numpy.transpose(data), model.ph) - numpy.dot(numpy.transpose(model.negdatastates), model.nh) )

                #pdb.set_trace()
                model.dc = numpy.array([numpy.sum(data, axis = 0) - numpy.sum(model.negdatastates, axis=0)])
                model.db = numpy.array([numpy.sum(model.ph, axis = 0) - numpy.sum(model.nh, axis = 0)])
                model.dWc = numpy.dot(numpy.transpose(classes), model.ph) - numpy.dot( numpy.transpose(model.negclassesstates), model.nh)
                model.dcc = numpy.array([numpy.sum(classes, axis = 0) - numpy.sum(model.negclassesstates, axis = 0)])
                model.Winc = numpy.dot(momentum, model.Winc) + numpy.dot(eta, (model.dW/numcases - numpy.dot(penalty, model.W)))
                model.binc = numpy.dot(momentum, model.binc) + numpy.dot(eta, (model.db/numcases))
                model.cinc = numpy.dot(momentum, model.cinc) + numpy.dot(eta, (model.dc/numcases))

                model.Wcinc = numpy.dot(momentum, model.Wcinc) + numpy.dot(eta, (model.dWc/numcases - numpy.dot(penalty, model.Wc)))
                model.ccinc = numpy.dot(momentum, model.ccinc) + numpy.dot(eta, (model.dcc/numcases))

                #pdb.set_trace()
                model.W = model.W + model.Winc
                model.b = model.b + model.binc
                model.c = model.c + model.cinc
                model.Wc = model.Wc + model.Wcinc
                model.cc = model.cc + model.ccinc

                if(epoch > avgstart):
                    model.Wavg = model.Wavg - numpy.dot((1/t), (model.Wavg-model.W))
                    model.cavg = model.cavg - numpy.dot((1/t), (model.cavg-model.c))
                    model.bavg = model.bavg - numpy.dot((1/t), (model.bavg-model.b))
                    model.Wcavg = model.Wcavg - numpy.dot((1/t), (model.Wcavg-model.Wc))
                    model.ccavg = model.ccavg - numpy.dot((1/t), (model.ccavg-model.cc))
                    t = t+1
                else:
                    model.Wavg = model.W
                    model.bavg = model.b
                    model.cavg = model.c
                    model.Wcavg = model.Wc
                    model.ccavg = model.cc
                err = numpy.sum(numpy.sum( (data-model.negdata)*(data-model.negdata), axis = 0 ), axis = 0)
                errsum = err + errsum

                #print "cur:%d batch training err:%f"%(batch,err)
            print "reconstruct error:%f"%errsum
            self.errors[0][epoch] = errsum
        model.W  = model.Wavg
        model.b = model.bavg
        model.c = model.cavg
        model.Wc = model.Wcavg
        model.cc = model.ccavg
        tmp = list(self.in_params['u_classes'])
        model.labels = numpy.transpose(numpy.array([tmp]))
        return model



