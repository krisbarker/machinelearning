#!/usr/bin/env python3

import sys
from os import system
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
from sklearn.datasets import load_iris
# import time


def initial_play():
    print("This is where we start with the tests")
    iris_dataset = load_iris()
    print("Keys of iris_dataset: \n", iris_dataset.keys())
    # print(iris_dataset['DESCR'] + "\n...")
    print("Target names:", iris_dataset['target_names'])
    print("Feature names: \n", iris_dataset['feature_names'])
    print("Type of data:", type(iris_dataset['data']))
    print("Shape of the data:", iris_dataset['data'].shape)
    print("The first five rows of data:\n", iris_dataset['data'][:5])
    print("Type of target:", type(iris_dataset['target']))
    print("Shape of target:", iris_dataset['target'].shape)
    print("Target:\n", iris_dataset['target'])


def initial_prints():
    print("Python version: ", sys.version)
    print("pandas version: ", pd.__version__)
    print("matplotlib version: ", matplotlib.__version__)
    print("NumPy version: ", np.__version__)
    print("SciPy version: ", sp.__version__)
    print("IPython version: ", IPython.__version__)
    print("scikit-learn version: ", sklearn.__version__)


def main():
    initial_prints()
    # time.sleep(2)
    system('clear')
    initial_play()


if __name__ == "__main__":
    main()
