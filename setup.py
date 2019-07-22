from setuptools import setup, find_packages

setup(
    name="Recourse",
    version="0.1.1",
    packages=find_packages(),
    install_requires=open('requirements.txt').read().split('\n')
)