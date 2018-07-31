# SageMaker Tutorials

This is the code accompanying the following Medium blogposts:

[Getting Started with SageMaker](https://medium.com/apache-mxnet/getting-started-with-sagemaker-ebe1277484c9)

[Leveling Up on SageMaker](https://medium.com/apache-mxnet/leveling-up-on-sagemaker-c7a5a438f0f6)

[94% Accuracy on Cifar-10 in 10 minutes on SageMaker](https://medium.com/apache-mxnet/94-accuracy-on-cifar-10-in-10-minutes-with-amazon-sagemaker-754e441d01d7)


# Debuging SageMaker

Currently, debugging SageMaker is not an ideal task. The easiest way to get get through, is by first creating the right conda environment to be used by SageMaker:

```
conda create -n debug_sage python=2.7
source activate debug_sage
pip install sagemaker-container-support
pip install pip install mxnet-cu90==1.1.0 # remove "-cu90" for CPU-only support
conda install ipykernel
source deactivate
```

Once the environment is created, it might take a few minutes so that the Jupyter Notebook recognizes it as a kernel. At which point, we can choose "debug_sage" as our kernel and run the training script by providing the right parameters:

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

