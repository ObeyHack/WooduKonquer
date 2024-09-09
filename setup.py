

from setuptools import setup, find_packages

setup(
    name="WooduKonquer",
    version="0.1.0",
    author="helpingstar",
    author_email="iamhelpingstar@gmail.com",
    description = "Search and Reinforcement Learning algorithms for the game of Woodchuck",
    license="MIT License",
    packages=find_packages(),
    install_requires=[
        "git+https://github.com/helpingstar/gym-woodoku.git",
    ],
    python_requires=">=3.9",
)