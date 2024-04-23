from setuptools import setup


setup(
    name='difformer',
    version="0.1.0",
    packages=["difformer"],
    install_requires=[
        "numpy<1.21.0",
        "fairseq==0.10.2",
        "sacrebleu<2.0.0",
        "sacremoses",
        "bert-score",
        "tensorboard",
        "tensorboardX",
    ],
)