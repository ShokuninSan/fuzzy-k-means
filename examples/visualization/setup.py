# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name = "plotting",
    version = "0.1.",
    author = "René Blaim",
    author_email = "me@flatmap.io",
    description = "Module for plotting Iris data via Plotly",
    license = "BSD",
    keywords = "iris data plotly",
    packages = find_packages(),
    install_requires = [
        "plotly",
        "pandas"
    ]
)
