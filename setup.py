from setuptools import setup, find_packages

setup(
    name="conversation_analytics",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'textblob',
        'nltk',
        'sentence-transformers'
    ],
) 