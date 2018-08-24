# SageMaker Tutorials

This is the code accompanying the following Medium blogposts:

[Getting Started with SageMaker](https://medium.com/apache-mxnet/getting-started-with-sagemaker-ebe1277484c9)

[Leveling Up on SageMaker](https://medium.com/apache-mxnet/leveling-up-on-sagemaker-c7a5a438f0f6)

[94% Accuracy on Cifar-10 in 10 minutes on SageMaker](https://medium.com/apache-mxnet/94-accuracy-on-cifar-10-in-10-minutes-with-amazon-sagemaker-754e441d01d7)


# Debuging SageMaker

Currently, debugging SageMaker is not an ideal task. The easiest way to get get through, is by first creating the right conda environment to be used by SageMaker. To do so, simply download the *setup.py* and the *debug_setup.sh* files. Afterwards, simply run the bash script with an argument indicating wether you want to install GPU support or CPU support:

```python
bash debug_setup.sh gpu
```


Once the environment is created, it might take a few minutes so that the Jupyter Notebook recognizes it as a kernel. At which point, we can choose it as our kernel and run the training script by providing the right parameters:

```python
from multiprocessing import cpu_count
from mxnet.test_utils import list_gpus
current_host='debug_algo'
hosts=[current_host]
num_cpus=cpu_count()
num_gpus=len(list_gpus())
channel_input_dirs={'training':'data'}
model_dir='./'
hyperparameters={'batch_size': 128, 
                  'epochs': 40}
                  
from source_dir_res18.run import train  # import the train function from 
                                        # the entrypoint that launches the training loop
train(current_host, hosts, num_cpus, num_gpus, channel_input_dirs, model_dir, hyperparameters)

```

Debugging works with **pdb** by setting a trace in the training loop. Once you run the training script, you will get an interactive session with the script.


## Using MXBoard

The abiblity to use MXBoard is, as of August 24 2018, not supported. To use this functionnality, we need to write our own `estimator.py` code, which is available here. You need to use this file to replace the `sagemaker/mxnet/estimator.py`  file contained in your sagemaker installion.

Since MXBoard is not installed by default on the Docker image used to train your job, you also need to include.

# Building your docker image
You can do this either by modifying this [file](https://github.com/aws/sagemaker-mxnet-container/blob/master/docker/1.1.0/final/Dockerfile.gpu) (i.e. by adding the `pip install mxboard` line to it). After which you can run the line:

`docker build -t preprod-mxnet:1.1.0-cpu-py2 --build-arg py_version=2
--build-arg framework_installable=mxnet-1.1.0-py2.py3-none-manylinux1_x86_64.whl -f Dockerfile.cpu .`

To run the previous line, you also need the `sagemaker_mxnet_container-1.0.0.tar.gz` file and the `mxnet-1.1.0-py2.py3-none-manylinux1_x86_64.whl`. For the former, you can get it by following intructions [here](https://github.com/aws/sagemaker-mxnet-container/issues/25). For the latter, you simply to fetch from [here](https://pypi.org/project/mxnet/1.1.0/#files). You can then proceed to upload your image on the internet and use it with your estimator through the `image_name` argument.


# Using our image

You can also use our image which comes with MXNet 1.1.0 and MXBoard: 968277166688.dkr.ecr.us-east-1.amazonaws.com/autoaugment:latest

# Launching the job

You can now launch your training job and specify the flag `run_tensorboard_locally` to True:

```python
estimator = MXNetEstimator(entry_point='train.py', 
                           role=sagemaker.get_execution_role(),
                           train_instance_count=1, 
                           train_instance_type='ml.p3.2xlarge',
                           image_name='968277166688.dkr.ecr.us-east-1.amazonaws.com/autoaugment:latest',
                           hyperparameters={'batch_size': 1024, 
                                            'epochs': 30})
estimator.fit(inputs,run_tensorboard_locally=True)
```

That should do it. Now you can run the following command:

`tensorboard --logdir=./logs --host=localhost --port=6007`

where the port number will be indicated by your training script.
