from setuptools import setup, find_packages

setup(
    name="conversation_analytics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "psutil>=5.9.0",
    ],
    author="Zyra",
    description="A package for analyzing conversations using NLP and ML techniques",
    python_requires=">=3.8",
) 