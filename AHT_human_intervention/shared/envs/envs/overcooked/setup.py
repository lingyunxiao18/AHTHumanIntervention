#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="overcooked_ai",
    version="0.0.1",
    description="Cooperative multi-agent environment based on Overcooked",
    author="Anonymous",
    author_email="anonymous@example.com",
    packages=find_packages(),
    install_requires=["numpy", "tqdm", "gym"],
)
