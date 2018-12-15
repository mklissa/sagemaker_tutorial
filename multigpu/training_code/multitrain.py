from __future__ import print_function

import json
import logging
import os
import time

import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models

#additional imports
from cifar10 import Cifar10
from resnet_config import WideResNet
from converters import DataIterLoader
import random
import numpy as np


    
# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def train(current_host, hosts, num_cpus, num_gpus, 
          channel_input_dirs, model_dir, hyperparameters, **kwargs):
    
    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 128)
    epochs = hyperparameters.get('epochs', 100)
    momentum = hyperparameters.get('momentum', 0.9)
    wd = hyperparameters.get('wd', 0.0005)
    lr_multiplier = hyperparameters.get('lr_multiplier', 1.)
    lr_schedule = {0: 0.01*lr_multiplier, 5: 0.1*lr_multiplier, 95: 0.01*lr_multiplier, 110: 0.001*lr_multiplier}
    

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
            
    #Set-up the right context 
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]    

    #Set-up the right batch size
    batch_size *= max(1, len(ctx))
    
    

    # Prepare the data
    given_path = channel_input_dirs['training']
    train_data, valid_data = Cifar10(batch_size=batch_size,
                                     data_shape=(3, 32, 32),
                                     padding=4,
                                     padding_value=0,
                                     normalization_type="channel",
                                     given_path=given_path,
                                     num_parts=len(hosts),
                                     part_index=part_index,
                                     num_cpus=num_cpus).return_dataloaders()



    # Create the model
    model = WideResNet(num_classes=10, depth=40, k=2) # We will use WideResNet40-2
    model.initialize(mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2), ctx=ctx)
    
    optimizer = mx.optimizer.SGD(learning_rate=lr_schedule[0],
                                 rescale_grad=1.0/batch_size,
                                 momentum=momentum, wd=wd)
    
    trainer = mx.gluon.Trainer(params=model.collect_params(),
                               optimizer=optimizer,
                               kvstore=kvstore)

    train_metric = mx.metric.Accuracy()
    criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    early_stopping_criteria=lambda e: e >= 0.94

    start_time=time.time()
    for epoch in range(epochs):       
        epoch_tick = time.time()

        train_metric.reset()     

        # update learning rate according to the schedule
        if epoch in lr_schedule.keys():
            trainer.set_learning_rate(lr_schedule[epoch])
            logging.info("Epoch {}, Changed learning rate.".format(epoch))
        logging.info('Epoch {}, Learning rate={}'.format(epoch, trainer.learning_rate))


        for batch_idx, (data, label) in enumerate(train_data):
            batch_tick = time.time()
            batch_size = data.shape[0]

            data = mx.gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = mx.gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)

            y_pred = []
            losses = []
            with mx.autograd.record():
                for x_part, y_true_part in zip(data, label):
                    y_pred_part = model(x_part)
                    loss = criterion(y_pred_part, y_true_part)
                    # Calculate loss on each partition of data.
                    # Store the losses and do backward after we have done forward on all GPUs,
                    # for better performance on multiple GPUs.
                    losses.append(loss)
                    y_pred.append(y_pred_part)
                for loss in losses:
                    loss.backward()
            trainer.step(batch_size)
            train_metric.update(label, y_pred)


        # log training accuracy
        _, trn_acc = train_metric.get()
        logging.info('Epoch {}, Training accuracy={}'.format(epoch, trn_acc))

        # log validation accuracy
        _, val_acc = test(ctx, model, valid_data)
        logging.info('Epoch {}, Validation accuracy={}'.format(epoch, val_acc))

        logging.info('Epoch {}, Duration={}'.format(epoch, time.time() - epoch_tick))
        if early_stopping_criteria:
            if early_stopping_criteria(val_acc):
                logging.info("Epoch {}, Reached early stopping target, stopping training.".format(epoch))
                break

   
    return model


def test(ctx, net, test_data):
    metric = mx.metric.Accuracy()

    for i, (data, label) in enumerate(test_data):
        data = gluon.utils.split_and_load(data, ctx_list=ctx)
        label = gluon.utils.split_and_load(label, ctx_list=ctx)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()


def save(net, model_dir):
    net.save_params('%s/model.params' % model_dir)
    

def model_fn(model_dir):
    net = WideResNet(num_classes=10, depth=40, k=2)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    nda = mx.nd.array(json.loads(data))

    # Just as in the training loop,
    # let's perform normalization on new data.
    means= [0.4914*255,0.4822*255,0.4465*255]
    stds = [0.2023*255,0.1994*255,0.2010*255]
    for i in range (3):
        nda[:,i]-= means[i]
        nda[:,i]/=stds[i]
    
    output = net(nda)
    prediction = mx.nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist()[0])
    return response_body, output_content_type
