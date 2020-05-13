from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# lddong_description(後述)に、GitHub用のREADME.mdを指定
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='StealthFlow',
    packages=['stealthflow'],
    version='0.0.6',
    license='MIT',
    install_requires=[],
    author='StealthFlow',
    author_email='deeplearnerstealthflow@gmail.com',
    url='https://github.com/p-geon/StealthFlow',
    description='Supporting Modules for MachineLearning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='stealthflow StealthFlow',

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ], # パッケージ(プロジェクト)の分類。https://pypi.org/classifiers/に掲載されているものを指定可能。
)
