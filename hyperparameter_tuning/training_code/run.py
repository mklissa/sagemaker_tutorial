from __future__ import print_function

import json
import logging
import os
import time
import random
import numpy as np

import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon.data.vision import datasets, transforms
from mxnet.gluon.data import DataLoader

#additional imports
from mytransforms import pad, RandomCrop, Compose
from continuousbatch import ContinuousBatchSampler
from resnet18_basic import resnet18Basic
from gluon import GluonLearner


# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def train(current_host, hosts, num_cpus, num_gpus, channel_input_dirs, model_dir, hyperparameters, **kwargs):
    # Set random seeds
    random.seed(0)
    np.random.seed(0)
    mx.random.seed(777)

    ## retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 128)
    epochs = hyperparameters.get('epochs', 30)
    extra = hyperparameters.get('extra', 3)
    peak = hyperparameters.get('peak', 5)

    momentum = hyperparameters.get('momentum', 0.91)
    lr=hyperparameters.get('lr', 0.1)
    min_lr=hyperparameters.get('min_lr', 0.005)
    wd = hyperparameters.get('wd', 0.0005)
    ##


    #Inform properly the algorithm about hardware choice
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync'

    part_index = 0
    for i, host in enumerate(hosts):
        if host == current_host:
            part_index = i
            break
    num_parts=len(hosts)


    #Set-up the right context 
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]



    # Define the transforms to be applied
    means=[0.4914,0.4822,0.4465]
    stds= [0.2023,0.1994,0.2010]      
    transform_train = Compose(
        [
         RandomCrop(32, padding=4, padding_mode='reflect'), 
         transforms.RandomFlipLeftRight(),            
         transforms.ToTensor(), 
         transforms.Normalize(means,stds)
        ]
    )

    transform_test = Compose(
        [
         transforms.ToTensor(), 
         transforms.Normalize(means,stds)
        ]
    )    


    # Load the datasets
    given_path = channel_input_dirs['training']
    trainset = datasets.CIFAR10(root=given_path, train=True)
    testset = datasets.CIFAR10(root=given_path, train=False)
    sampler = mx.gluon.data.RandomSampler(len(trainset))
    batch_sampler = ContinuousBatchSampler(sampler,
                                           batch_size=batch_size,
                                           datalen=len(trainset),
                                           max_iters=int(len(trainset)/batch_size + 1)*45)    
    train_data = DataLoader( 
        trainset.transform_first(transform_train),
        batch_sampler=batch_sampler,
        num_workers=num_cpus,
    )
    valid_data = DataLoader(
        testset.transform_first(transform_test),
        batch_size=1024,
        shuffle=False,
        num_workers=num_cpus,
    )    
    
    
    

    # Define model and the dtype (wether to use full precision or half precision)
    dtype='float16'
    model = resnet18Basic(num_classes=10)

    
    # Define the learner and fit!
    learner = GluonLearner(model, hybridize=False, ctx=ctx)
    learner.fit(train_data=train_data, valid_data=valid_data,

                epochs=epochs, extra=extra, peak=peak,
                lr=lr, min_lr=min_lr,  momentum=momentum,

                initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2),
                optimizer=mx.optimizer.SGD(learning_rate=lr, rescale_grad=1.0/batch_size,
                                           momentum=momentum, wd=wd, multi_precision=(dtype=='float16')),
                early_stopping_criteria=lambda e: e >= 0.94,

                kvstore=kvstore, dtype=dtype,
               ) 

    
    
   
    return model


def save(net, model_dir):
    # save the model
    net.save_params('%s/model.params' % model_dir)
    pass
    
    
def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    
    model = resnet18Basic(num_classes=10)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    parsed = json.loads(data)
    nda = mx.nd.array(parsed)
    

    means= [0.4914*255,0.4822*255,0.4465*255]
    stds = [0.2023*255,0.1994*255,0.2010*255]
    for i in range (3):
        nda[:,i]-= means[i]
        nda[:,i]/=stds[i]
        
    
    output = net(nda)
    prediction = mx.nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist()[0])

    return response_body, output_content_type
