from setuptools import setup, find_packages

setup(
    name="ml_teammate",
    version="0.1",
    packages=find_packages(),  # <---- this line discovers ml_teammate/*
    install_requires=[],
)
