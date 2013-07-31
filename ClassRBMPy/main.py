#!coding=utf-8
"""
This code realize the basic version of classification RBM. Since my understanding about RBM may not deep enough, the code would exist many problems, so I strongly not recommand you using this code in practical. Thank you for your reading and best wishes.
"""
import gen_input
from loading_model import *
import pdb
import config
import rbm
import rbm_predict



if __name__ == "__main__":
    print "Start RBM Test..."
    print "Loading data and generate input layer format..."
    [data, labels, testdata, groundtruth] = gen_input.load('./data/mnist.pkl.gz')
    print "training RBM..."
    con = config.Config()
    con.set_data(data)
    con.set_labels(labels)
    rbm_obj = rbm.RBM(con)
    model = rbm_obj.train()
    save_model('./train_model/',model)
    yhat = rbm_predict.predict(model, testdata)
    groundtruthT = numpy.transpose(groundtruth)
    counter = 0
    for i in range(0, len(yhat)):
        if abs(groundtruthT[0][i]-yhat[i]) > 0.00001:
            counter += 1
            print "counter:%d num:%d  Error predict:%d  groundtruth%d"%(counter,i, int(yhat[i]), int(groundtruthT[0][i]))

