from setuptools import setup, find_namespace_packages

setup(
    name="PyToyTorch",
    version="0.1",
    author="mnicnc404",
    author_email="mnicnc404@gmail.com",
    packages=find_namespace_packages(include=("PyToyTorch.*")),
    install_requires=[
        'numpy',
    ],
)
