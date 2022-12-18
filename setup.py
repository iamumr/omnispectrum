#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='omnispectrum',
    version = 0.7,
    description=(
        '全向地震动反应谱生成程序'
    ),
    long_description=open('README.md', encoding = 'utf-8').read(),
    long_description_content_type="text/markdown",
    author = 'Guochang Li',
    author_email = 'liguochangli@gmail.com',
    url = 'http://umr.ink',
    license = 'Apache 3.0',
    packages = ['omnispectrum'],
    include_package_data=True,
    package_data={'omnispectrum':['*.csv']},
    platforms = ["all"],
    install_requires = [
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'tqdm',
        'python-docx'
    ],
)
