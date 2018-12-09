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
    entry_points=dict(console_scripts=[
        'ppo=ppo.main:cli',
        'hsr=ppo.main:hsr_cli',
        'tb=scripts.tensorboard:main',
    ]),
    install_requires=[
        'baselines==0.1.5',
        'Cython==0.29',
        'gym==0.10.9',
        'matplotlib==3.0.2',
        'numpy==1.15.4',
        'opencv-python==3.4.3.18',
        'torch==0.4.1',
    ])
