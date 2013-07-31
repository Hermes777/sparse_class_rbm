#!coding=utf-8
import pdb
import numpy


class Config:
    def __init__(self):
        self.params = {}
        self.params['X'] = None           # test data, format in binary.
        self.params['numhid'] = 100       # number of hidden units
        self.params['y'] = None           # labels list
        self.params['nclasses'] = None    # number of classes, how many classes in y
        self.params['method'] = "CD-k"    # training method
        self.params['eta'] = 0.1          # learning rate
        self.params['momentum'] = 0.5     # mementum for smoothness and to prevent overfitting.
        self.params['maxepoch'] = 50       # max loop for train, each loop is a full pass through train data
        self.params['avglast'] = 5        # how many epochs before maxepoch to start averaging. Procedure suggested for faster convergence by Kevin Swersky in his MSc thesis.
        self.params['penalty'] = 2e-4    # weight decay factor
        self.params['weightdecay'] = True # a bool flag. when set to true, the weights are decayed linerly from penalty->0.1*penality in epochs.
        self.params['batchsize'] = 100    # the number of training instances per batch.
        self.params['anneal'] = False     # a bool flag. if set true, the penalty is annealed linerly through epoch to 10% of its original value
        self.params['u_classes'] = set()  # unique value of output.

    def parse_config_file(self, con_file):
        """
        parsing config file.
        """
        fobj = open(con_file, 'r')
        for line in fobj:
            line = line.strip()
            if line.find("#")  == 0:
                continue
            tmp = line.split("=")
            try:
                if isinstance(tmp[1], int):
                    self.params[tmp[0]] = int(tmp[1])
                elif isinstance(tmp[1], float):
                    self.params[tmp[0]] = float(tmp[1])
                else:
                    self.params[tmp[0]] = tmp[1]
            except Exception, e:
                print "Error key, input:%s, key:%s, val:%s"%(line,tmp[0],tmp[1])
    def set_data(self, X):
        self.params['X'] = X

    def set_labels(self, y):
        self.params['y'] = y
        for i in range(0, numpy.size(y, 0)):
            self.params['u_classes'].add(y[i][0])
        self.params['nclasses'] = len(self.params['u_classes'])

    def get_params(self):
        return self.params

