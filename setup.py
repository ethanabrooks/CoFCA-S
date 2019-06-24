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
    scripts=[
        'bin/load',
        'bin/new-run',
        'bin/from-json',
        'bin/dbg',
        'bin/show-best',
        'bin/reproduce',
    ],
    entry_points=dict(console_scripts=[
        'ppo=ppo.main:cli',
        'subtasks=ppo.main:subtasks_cli',
        'control-flow=ppo.main:control_flow_cli',
        'teacher=ppo.main:teacher_cli',
        'student=ppo.main:student_cli',
    ]),
    install_requires=[])
