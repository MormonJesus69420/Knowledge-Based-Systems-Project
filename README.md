# Documentation
Documentation and general usage guide for project in STE6246 Knowledge Based Systems Project. Assignment 2 runs on Tensorflow-GPU insted of oridinary Tensorflow as it allows me to cut on training time by a factor of 10 for each epoch. So instead of using 60 seconds per epoch I am only using 6.

## Prerequisites
### Python
 * Python 3.7.0 or newer is required to run Assignment 1. [Python](https://www.python.org/)
 * Python 3.6.6 or **later** is required to run Assignment 2 (Tensorflow does not support Python 3.7 as of 2018-10-26). [Python](https://www.python.org/)
 * CUDA 9.0 **exactly** is required to run Assignment 2. (Required by Tensorflow on GPU). [CUDA](https://developer.nvidia.com/cuda-zone)
 * cuDNN 7.2.1 **exactly** is required to run Assignment 2 (Required by Tensorflow on GPU). [cuDNN](https://developer.nvidia.com/cudnn)
 * pyenv for managing separate versions of Python on same pc. [pyenv](https://github.com/pyenv/pyenv)
 * pyenv-virtualenv for managing virtual environments. [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)

### Libraries
These libraries are necessary for my project, they are installed in the [Setting Up Code](#Setting-Up-Code).
 * Keras 2.2.4 or newer [PyPI page for Keras](https://pypi.org/project/Keras/)
 * matplotlib 3.0.0 or newer [PyPI page for Matplotlib](https://pypi.org/project/matplotlib/)
 * mumpy 1.15.3 or newer: [PyPI page for Numpy](https://pypi.org/project/numpy/)
 * scikit-learn 0.20.0 or newer [PyPI page for scikit-learn](https://pypi.org/project/scikit-learn/)
 * scipy 1.1.0 or newer [PyPI page for scipy](https://pypi.org/project/scipy/)
 * tensorflow-gpu 1.11.0 or newer [PyPI page for tensorflow-gpu](https://pypi.org/project/tensorflow-gpu/)

## Running The Project 
### Setting Up Code
In order to run the code you must first run file SETUP.sh or execute following commands in root directory of the project:
```bash
# Install Python 3.7 & 3.6
pyenv install 3.7.0
pyenv install 3.6.6

# Create virtual environments
pyenv virtualenv 3.7.0 venv-3.7
pyenv virtualenv 3.6.6 venv-3.6

# Activate virtual environment (repeat for 3.6 after last step)
pyenv activate venv-3.7

# Upgrade pip to newest version
pip3 install --upgrade pip setuptools

# Download all required packages from requirements.txt
pip3 install -r requirements.txt

# Deactivate environment
pyenv deactivate
```
This will setup the virtual environments and install necessary packages and their requirements so that you can run my project.

### Running files
If you have properly installed pyenv and pyenv-virtualenv properly installed it should automatically activate appropriate environment based on project requirement. If that does not happen, activate necessary venv by running:
```bash
pyenv activate venv-3.7 {or venv-3.6}
```
After you have activated the virtual environment you can navigate with your terminal to a project you want to run, for example:
```bash
cd ./Assignment1/Part1
```
And then run the python script using following command:
```bash
python3 Simulation1.py
```
This will run the python code in safe environment, separate from your personal Python installation, and with required packages already setup.

### Finishing work
To close the virtual environment simply end your terminal session or run `pyenv deactivate` command in your terminal.

## Author
**Daniel Aaron Salwerowicz** - *Developer and memer* -
[CodeRefinery](https://source.coderefinery.org/MormonJesus69420)

## Acknowledgments
**Christopher Kragebøl Hagerup** - *Developer and weeb* -
[CodeRefinery](https://source.coderefinery.org/Krahager)  
**Kent Arne Larsen** - *Developer and boomer* -
[CodeRefinery](https://source.coderefinery.org/kla096)  
**Hans Victor Andersson Lindbäck** - *Developer and svenskefæn* -
[CodeRefinery](https://source.coderefinery.org/hli039)  
**Olav Kjartan Larseng** - *Developer and c-menneske* -
[CodeRefinery](https://source.coderefinery.org/ola014)
