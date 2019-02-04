#! /usr/bin/env python

# third party
from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='on-policy-curiosity',
    version='0.0.0',
    long_description=long_description,
    url='https://github.com/lobachevzky/on-policy-curiosity',
    author='Ethan Brooks',
    author_email='ethanabrooks@gmail.com',
    packages=find_packages(),
    scripts=['load'],
    entry_points=dict(console_scripts=[
        'ppo=ppo.main:cli',
        'unsupervised=ppo.main:unsupervised_cli',
        'hsr=ppo.main:hsr_cli',
        'unsupervised-hsr=ppo.main:unsupervised_hsr_cli',
    ]),
    install_requires=[
        'gym',
        'numpy==1.15.4',
        'opencv-python==3.4.3.18',
        'torch==1.0.0',
        'tensorboardX==1.6',
        'joblib==0.13.1'
    ])
