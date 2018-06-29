from __future__ import print_function

import json
import logging
import os
import time

import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data.vision import transforms

def train(channel_input_dirs, hyperparameters):
    
    # Retrieve the hyperparameters
    batch_size = hyperparameters.get('batch_size', 55)
    epochs = hyperparameters.get('epochs', 100)
    ctx=[mx.gpu()]
    
    
    # Prepare the data
    data_dir = channel_input_dirs['training']
    train_data = get_data(data_dir, batch_size, train=True)
    test_data = get_data(data_dir, batch_size, train=False)

    
    # Create the model
    net = models.get_model('resnet34_v2', ctx=ctx, pretrained=False, classes=10)
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            optimizer_params={'learning_rate': 0.05,
                                              'momentum': 0.9,
                                              'wd':1e-4})
    metric = mx.metric.Accuracy()
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()



    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        train_data.reset()
        tic = time.time()
        metric.reset()

        for i, batch in enumerate(train_data):
            #Load the data on GPU
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            losses = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    loss = criterion(z, y)
                    losses.append(loss)
                    outputs.append(z)
                for loss in losses:
                    loss.backward()
            trainer.step(batch_size)
            metric.update(label, outputs)

        name, acc = metric.get()
        logging.info('[Epoch %d] training: %s=%f' % (epoch, name, acc))
        logging.info('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))

        name, val_acc = test(ctx, net, test_data)
        logging.info('[Epoch %d] validation: %s=%f' % (epoch, name, val_acc))

    return net


def get_data(data_dir, batch_size, data_shape=(3,32,32), train=True):

    path = data_dir + "/train.rec" if train else data_dir + "/test.rec"

    return mx.io.ImageRecordIter(
        path_imgrec=path,
        data_shape=data_shape,
        batch_size=batch_size,
        rand_crop=train,
        rand_mirror=train)


def test(ctx, net, test_data):
    test_data.reset()
    metric = mx.metric.Accuracy()

    for i, batch in enumerate(test_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()



def save(net, model_dir):
    net.save_params('%s/model.params' % model_dir)

    
def model_fn(model_dir):
    net = models.get_model('resnet34_v2', ctx=mx.cpu(), pretrained=False, classes=10)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    inputs = mx.nd.array(json.loads(data))
    outputs = net(inputs)
    predictions = mx.nd.argmax(outputs, axis=1)
    response = json.dumps(predictions.asnumpy().tolist()[0])
    return response, output_content_type
