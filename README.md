# Code With Me _ Machine Learning
Welcome to Code With Me Machine Learning Night. We will install Python, TensorFlow, and learn how to build simple Neural Networks, and how they work.

# Installing Python 
We want to install Python 3.6 (64 bit version). We can find the latest version at Python's main website:

https://www.python.org/downloads/release/python-363/

We can see the different OS versions as well. For windows it is recommended that Python is installed in a new folder ex) C:\Python

## During Instillation:
It is recommend that all check boxes are checked, specifically "add Python to PATH" for Windows.

## Ensure Python is fully installed and works.
In the command line or terminal, typing ```python``` or ```python3``` should open the Python Interpreter in the command line or terminal. You can perform quick calculations to make sure it works.

```>>> 2+2```

```>>> print("Hello World")```

Typing ```>>> quit()``` exits the Python Interpreter.

# Installing PyCharm 
PyCharm is a full IDE. We can download the free community edition here:

https://www.jetbrains.com/pycharm/

# Pip and installing other Python Modules 
To install other Python modules we will use pip. In the command line or terminal we can type:

```> pip install [name_of_module]```

The first two modules we will install are wheel and ipython.

```> pip install wheel``` 

```> pip install ipython```

We can take a look at all the modules pip has installed by calling:

```> pip list```

# TensorFlow and the NumPy Stack
BEFORE we install TensorFlow we need to install the NumPy stack. Specifically on Windows ```> pip install numpy``` MIGHT NOT work with certain modules that depend on Numpy, therefore we should install ```numpy+mkl```. We can find the wheel file for the numpy+mkl instillation here at the un-offical python package index:

https://www.lfd.uci.edu/~gohlke/pythonlibs/

We can see in the heading paragraph that numpy+mkl-1.13 is used by many modules. For Windows the specific module we want to download is:

```numpy‑1.13.3+mkl‑cp36‑cp36m‑win_amd64.whl```

Numpy+MKL Version 1.13, CPython 3.6, Windows 64 bit. After the download we can open the command prompt and chage directory into the Downloads folder.

```> cd Downloads```

```> pip install numpy‑1.13.3+mkl‑cp36‑cp36m‑win_amd64.whl```

## Installing Machine Learning Libraries and the NumPy Stack
Now that we have NumPy+MKL installed, we can now install all the other modules we need.

```> pip install tensorflow pandas matplotlib scipy keras```

## Hello World in TensorFlow Example
After configuring the correct path to the Python 3.6.exe in PyCharm we can write our first script using TensorFlow!

```
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```















