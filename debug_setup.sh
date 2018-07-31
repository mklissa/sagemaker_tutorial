
if [ "$1" == "cpu" ];then
    echo "Installing for CPU"
elif [ "$1" == "gpu" ];then
    echo "Installing for GPU"
    echo "debug_sage_$1"
else
    echo "Unknown argument, exiting now."
    exit 1
fi

envname="debug_sage_$1"

conda create -y -n $envname python=2.7
source activate $envname
python setup.py install
if [ "$1" == "cpu" ];then
    pip install mxnet-mkl==1.1.0
elif [ "$1" == "gpu" ];then
    pip install mxnet-cu90==1.1.0
fi
conda install -y ipykernel
source deactivate


