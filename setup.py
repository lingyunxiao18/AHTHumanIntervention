#!/usr/bin/env python

import os

import setuptools
from setuptools import setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("AHT_human_intervention", "__init__.py")).read().split()
    return init[init.index("__version__") + 2][1:-1]


setup(
    name="aht-human-intervention",
    version=get_version(),
    description="HINT-Agent: human-in-the-loop multi-agent coordination demos",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "gym==0.22",
        "pygame",
        "tqdm",
        "openai",
        "jsonschema",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="human-in-the-loop multi-agent coordination overcooked crowdnav llm",
    python_requires=">=3.8",
)
