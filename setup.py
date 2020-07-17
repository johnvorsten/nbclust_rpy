# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:37:16 2020

@author: johnvorsten
vorstenjohn@gmail.com
"""

# Python imports
import setuptools

#%%

with open('README.md', 'r') as f:
    long_description = f.read()

short_description = """RPy2 interface to R Package NbClust"""

setuptools.setup(name="nbclust-rpy",
                 version="0.0.1",
                 author="John Vorsten",
                 author_email="vorstenjohn@gmail.com",
                 description=short_description,
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://https://github.com/johnvorsten/nbclust_rpy",
                 packages=setuptools.find_packages(),
                 license='MIT',
                 keywords='nbclust rpy R clustering',
                 classifiers=[
                     'Development Status :: 3 - Alpha',
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires='>=3',
                 install_requires=[
                     'rpy2',
                     'pandas',
                     'numpy',
                 ])
