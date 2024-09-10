from setuptools import setup, find_packages

setup(
    name="woodokuMind",
    version="0.1.0",
    author="NML",
    description = "Search and Reinforcement Learning algorithms for the game of WooduK",
    license="MIT License",
    packages=find_packages(),
    install_requires=[
        "git+https://github.com/helpingstar/gym-woodoku.git",
        "numpy",
        "tqdm",
        "keyboard",
        "python-dotenv",
        "seaborn",
        "matplotlib",
        "neptune",
        "opencv-python",
    ],
    python_requires=">=3.10",
)