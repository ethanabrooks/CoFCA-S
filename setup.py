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
        'hsr=ppo.main:hsr_cli',
        'logic=ppo.main:logic_cli',
        'teach=ppo.main:teach_cli',
    ]),
    install_requires=[
        'gym',
        'numpy',
        'torch',
        'tensorboardX==1.6',
    ])
