import os
import logging
import mxnet as mx
# from multiprocessing import cpu_count

from converters import DataIterLoader


class Cifar10():
    def __init__(self,
                 batch_size=128,
                 data_shape=(3,32,32),
                 padding=0,
                 padding_value=0,
                 normalization_type=None,
                 given_path=None,
                num_parts=1,
                part_index=0,
                num_cpus=1):
        """

        Parameters
        ----------
        batch_size : int
        data_shape
        padding : int
            Number of pixels to pad on each side (top, bottom, left and right)
        padding_value : int
            Value for padded pixels
        normalization_type : str, optional
            Should be either "pixel" or "channel"

        """

            
        self.data_path =given_path
        self.prepare_iters(batch_size, data_shape, normalization_type, padding, padding_value, num_parts ,part_index, num_cpus)

    def prepare_iters(self, batch_size, data_shape, normalization_type, padding,
                      padding_value, train_num_parts, train_part_index, num_cpus):

        shared_args = {'data_shape': data_shape,
                       'batch_size': batch_size,
                      'num_parts': 1,
                      'part_index':0,
                      'preprocess_threads': num_cpus}



        shared_args.update({
            'mean_r': 0.4914*255,
            'mean_g': 0.4822*255,
            'mean_b': 0.4465*255,
            'std_r': 0.2023*255,
            'std_g': 0.1994*255,
            'std_b': 0.2010*255
        })
            

        self.train_iter_args = shared_args.copy()
        self.train_iter_args.update({
            'path_imgrec': os.path.join(self.data_path, "cifar/train.rec"),
            'shuffle': True,
            'rand_crop': True,
            'rand_mirror': True,
            'pad': padding,
            'fill_value': padding_value,
            'num_parts':train_num_parts,
            'part_index':train_part_index
        })
        
        self.train_iter = mx.io.ImageRecordIter(**self.train_iter_args)

        self.test_iter_args = shared_args.copy()
        self.test_iter_args.update({'path_imgrec': os.path.join(self.data_path, "cifar/test.rec")})
        self.test_iter = mx.io.ImageRecordIter(**self.test_iter_args)

    def return_dataloaders(self):
        return DataIterLoader(self.train_iter), DataIterLoader(self.test_iter)

    def return_dataiters(self):
        return self.train_iter, self.test_iter