import logging
import time
import os
import sys
import mxnet as mx
from mxnet import nd



class GluonLearner():
    def __init__(self, model, ctx=[mx.cpu()]):

        logging.info("Using Gluon Learner.")
        self.model = model
        self.context = ctx
    

    def fit(self, train_data, valid_data,
            epochs=300, extra=7, peak=15, burnout=5,
            lr=None, momentum=None, min_lr=None, 
            initializer=mx.init.Xavier(),
            optimizer=None,
            kvstore='device',
            early_stopping_criteria=None,
            dtype='float32',         
        ):



        def linear_cycle(lr_initial=0.1, epochs=10, peak=5, low_lr=0.005, extra=5):
            def f(progress):
                if progress < peak:
                    return lr_initial * (float(progress) / peak) 
                elif progress <= epochs:
                    return low_lr + lr_initial - lr_initial * float(progress - peak) / (epochs-peak)
                elif progress <= epochs + extra:
                    return low_lr * float(extra - (progress - epochs)) / extra + low_lr / 10
                else:
                    return low_lr / 10
            return f

        def mom_linear_cycle(mom_initial=0.93, peak_mom=0.88, epochs=10, peak=5, extra=5):
            def f(progress):
                if progress < peak:
                    return mom_initial - (mom_initial-peak_mom) * float(progress) / peak
                elif progress <= epochs:
                    return peak_mom +  (mom_initial-peak_mom) * float(progress - peak) / (epochs-peak) 
                else:
                    return mom_initial
            return f   

        lr_scheduler = linear_cycle(lr_initial=lr, low_lr=min_lr, epochs=epochs, peak=peak, extra=extra)
        mom_scheduler = mom_linear_cycle(mom_initial=momentum, epochs=epochs, peak=peak, extra=extra)        
        
               

        self.model.initialize(initializer, ctx=self.context)
        self.model.cast(dtype) # Cast to float16 if needed.
        
        trainer = mx.gluon.Trainer(params=self.model.collect_params(), optimizer=optimizer, kvstore=kvstore)
        train_metric = mx.metric.Accuracy()
        criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        max_val_acc = {'val_acc': 0, 'trn_acc': 0, 'epoch': 0}

        batch_size=None
        first_batch=True
        epoch=0
        train_metric.reset()
        batch_tick = time.time()

   
        for batch_idx, (data, label) in enumerate(train_data):
            batch_size = data.shape[0]
            if first_batch: # To prevent batch size changes due to last batch being of different size.
                orig_batch_size = int(batch_size) # Used for learning rate and momentum schedules.
                first_batch=False


            #Update the learning rate and momentum
            new_lr = lr_scheduler(epoch + float(batch_idx % (50000/orig_batch_size + 1)) / int(50000/orig_batch_size))
            trainer.set_learning_rate(new_lr)
            new_mom = mom_scheduler(epoch + float(batch_idx % (50000/orig_batch_size + 1)) / int(50000/orig_batch_size))
            trainer._optimizer.momentum = new_mom

            # partition data across all devices in context
            data = mx.gluon.utils.split_and_load(data.astype(dtype), ctx_list=self.context, batch_axis=0) # Astype is used for float16.
            label = mx.gluon.utils.split_and_load(label, ctx_list=self.context, batch_axis=0)


            # Calculate loss and perform backward pass.
            y_pred = []
            losses = []
            with mx.autograd.record():
                # calculate loss on each partition of data
                for x_part, y_true_part in zip(data, label):
                    y_pred_part = self.model(x_part)
                    loss = criterion(y_pred_part, y_true_part)
                    if epoch ==0 and batch_idx ==0:  # We start the experiment time here, as just not to include
                        experiment_start=time.time() # the performance tests happening in the run feed-forward.
                        epoch_tick = time.time()  
                    losses.append(loss)
                    y_pred.append(y_pred_part)
                for loss in losses:
                    loss.backward()
            trainer.step(batch_size)
            train_metric.update(label, y_pred)
                

            if (batch_idx + 1) % (50000/orig_batch_size + 1) == 0 and batch_idx != 0: # Due to continuous sampling, 
                                                                                      # we have to manually check for the end of an epoch.



                # log training and validation accuracy
                _, trn_acc = train_metric.get()
                val_acc = evaluate_accuracy(valid_data, self.model, ctx=self.context,dtype=dtype)
                logging.info('Epoch {}, Validation accuracy={} , Train acc={}'.format(epoch, val_acc, trn_acc))      


                # log best job until now
                if val_acc > max_val_acc['val_acc']:
                    max_val_acc = {'val_acc': val_acc, 'trn_acc': trn_acc, 'epoch': epoch}
                logging.info("Epoch {}, (Max val={}, Max train={})".format(epoch,
                                                                            max_val_acc['val_acc'],
                                                                            max_val_acc['trn_acc'],)
                                                                            )

                # Log time
                logging.info("Epoch Time= {}, Tot Time = {}".format(time.time()-epoch_tick, time.time()-experiment_start))


                if early_stopping_criteria: # Stop after we achieved .94
                    if early_stopping_criteria(val_acc):
                        logging.info("Epoch {}, Reached early stopping target, stopping training.".format(epoch))
                        break
                epoch+=1        
                first_batch=True
                epoch_tick = time.time()


class WaitOnReadAccuracy():
    def __init__(self, ctx,dtype):
        self.dtype=dtype
        if isinstance(ctx, list):
            self.ctx = ctx[0]
        else:
            self.ctx = ctx
        self.metric = mx.nd.zeros(1, self.ctx,dtype=self.dtype)
        self.num_instance = mx.nd.zeros(1, self.ctx,dtype=self.dtype)

    def reset(self):
        self.metric = mx.nd.zeros(1, self.ctx,dtype=self.dtype)
        self.num_instance = mx.nd.zeros(1, self.ctx,dtype=self.dtype)

    def get(self):
        return float(self.metric.asscalar()) / float(self.num_instance.asscalar())

    def update(self, label, pred):
        # for single context
        if isinstance(label, mx.nd.NDArray) and isinstance(pred, mx.nd.NDArray):
            pred = mx.nd.argmax(pred, axis=1)
            self.metric += (pred == label).sum()
            self.num_instance += label.shape[0]
        # for multi-context where data is partitioned
        elif isinstance(label, list) and isinstance(pred, list):
            for label_part, pred_part in zip(label, pred):
                pred_part = mx.nd.argmax(pred_part, axis=1)
                self.metric += (pred_part == label_part).sum()
                self.num_instance += label_part.shape[0]
        else:
            raise TypeError


def evaluate_accuracy(valid_data, model, ctx,dtype):
    if isinstance(ctx, list):
        ctx = ctx[0]
    accuracy = WaitOnReadAccuracy(ctx,dtype)
    for batch_idx, (data, label) in enumerate(valid_data):     
        data = data.as_in_context(ctx).astype(dtype)
        label = label.as_in_context(ctx)
        output = model(data)
        accuracy.update(label.astype(dtype), output)
    return accuracy.get()

