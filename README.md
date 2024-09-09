# WooduKonquer

## How to run the project

### Environment Setup
To run the project, you need to have `python:3.10` or higher installed on your machine.
1. Create a virtual environment
```bash
python -m venv venv
```
2. Activate the virtual environment
```bash
source venv/bin/activate
```
3. Install the dependencies
```bash
pip install -r requirements.txt
```
4. Run the project
```bash
python src/wodukonquer.py -h
```





# Docker Usage
download docker desktop (from software center)
you may need to install WSL2 if it ask you to

## Building the image 
```bash
docker build . -t woodukonquer
```