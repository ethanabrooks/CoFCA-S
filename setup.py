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
        "bin/fix_config",
        "bin/load",
        "bin/search",
        "bin/seeds",
        "bin/sweep",
    ],
)
