

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages




setup(
    name='sagemaker_mxnet_container',
    version='1.0.0',
    description='Debugging SageMaker',

    install_requires=['sagemaker-container-support >= 1.0.0, <2','sagemaker','botocore','pandas'],


)