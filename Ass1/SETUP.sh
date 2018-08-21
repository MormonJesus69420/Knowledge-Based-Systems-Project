#!/bin/bash

# Setup virtual environment
python3 -m venv venv

# Activate virtual environment
source ./venv/bin/activate

# Upgrade pip to newest version
pip3 install --upgrade pip setuptools

# Download all required packages from requirements.txt
pip3 install -r requirements.txt
