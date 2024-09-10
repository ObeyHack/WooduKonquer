# WooduKonquer

Welcome to the WooduKonquer project, this project aims to explore different AI driven strategies to 
play the game of Woodoku

https://github.com/user-attachments/assets/444c8485-bcf0-4392-9b06-90f6459615cf



## How to run the project

### Environment Setup
To run the project, you need to have `python:3.10` or higher installed on your machine.
1. `clone` the project
```bash
git clone ...
```
2. Create a virtual environment
```bash
python -m venv venv
```
2. Activate the virtual environment
```bash
source venv/bin/activate
```
3. Install the dependencies
```bash
pip install .
```
4. Run the project
```bash
woodoku-cli -h
```





# Docker Usage
download docker desktop (from software center)
you may need to install WSL2 if it ask you to

## Building the image 
```bash
docker build . -t woodukonquer
```
