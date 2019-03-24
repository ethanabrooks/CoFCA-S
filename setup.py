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
        'tasks=ppo.main:tasks_cli',
        'hsr=ppo.main:hsr_cli',
        'tasks-hsr=ppo.main:tasks_hsr_cli',
        'save-state=ppo.main:save_state_cli',
    ]),
    install_requires=[
        'gym',
        'numpy',
        'opencv-python==3.4.3.18',
        'torch==1.0.0',
        'tensorboardX==1.6',
    ])
