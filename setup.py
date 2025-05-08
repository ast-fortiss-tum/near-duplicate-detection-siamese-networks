from setuptools import setup, find_packages

setup(
    name="ndd",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.30.0",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn==1.5.1",
        "tensorflow",
        "hyperopt",
        "natsort",
    ],
)
