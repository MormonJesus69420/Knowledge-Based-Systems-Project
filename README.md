# Documentation
Documentation and general usage guide for project in STE6246 Knowledge Based Systems Project. 

## Prerequisites
### Python
 * Python 3.7.0 or newer is required to run this project. [Python](https://www.python.org/)

### Libraries
These libraries are necessary for my project, they are installed in the [Setting Up Code](#Setting-Up-Code).
 * Numpy 1.14.4 or newer: [PyPI page for Numpy](https://pypi.org/project/numpy/)
 * Matplotlib 2.2.2 or newer [PyPI page for Matplotlib](https://pypi.org/project/matplotlib/)

## Running The Project 
### Setting Up Code
In order to run the code you must first run file SETUP.sh or execute following commands in root directory of the project:
```bash
python3 -m venv venv
source ./venv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt
```
This will setup the virtual environment and install necessary packages and their requirements so that you can run my project.

### Running files
If you have manually installed the packages as instructed in [Setting Up Code](#Setting-Up-Code) you should still have the virtual environment activated, if not run the following command in root directory of the project:
```bash
source ./venv/bin/activate
```
After you have activated the virtual environment you can navigate your terminal to a project you want to run, for example:
```bash
cd ./Assignment1/Part1
```
And then run the python script using following command:
```bash
python3 Simulation.py
```
This will run the python code in safe environment, separate from your personal Python installation, and with required packages already setup.

### Finishing work
To close the virtual environment simply end your terminal session or run `deactivate` command in your terminal.

## Author
**Daniel Aaron Salwerowicz** - *Developer and memer* -
[CodeRefinery](https://source.coderefinery.org/MormonJesus69420)

## Acknowledgments
**Christopher Kragebøl Hagerup** - *Developer and weeb* -
[CodeRefinery](https://source.coderefinery.org/Krahager)
