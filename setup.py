#! /usr/bin/env python

# third party
from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="on-policy-curiosity",
    version="0.0.0",
    long_description=long_description,
    url="https://github.com/lobachevzky/on-policy-curiosity",
    author="Ethan Brooks",
    author_email="ethanabrooks@gmail.com",
    packages=find_packages(),
    scripts=[
        "bin/load",
        "bin/load1",
        "bin/new-run",
        "bin/from-json",
        "bin/dbg",
        "bin/dbg1",
        "bin/show-best",
        "bin/reproduce",
    ],
    entry_points=dict(
        console_scripts=[
            "oh-et-al=ppo.oh_et_al.main:cli",
            "picture-hanging=ppo.picture_hanging.main:cli",
        ]
    ),
    install_requires=[
        "ray[debug]",
        "tensorboardX",
        "tensorflow",
        "opencv-python",
        "psutil",
        "requests",
    ],
)
