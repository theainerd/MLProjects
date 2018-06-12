# Convolutional Neural Network for Keyword Spotting
I have used the Speech Commands Dataset to build an algorithm that understands simple spoken commands.

### Install

This project requires **Python3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included

## Project Structure
- data
	- raw
		- train (Training audio files)
		- test (Test audio files used for evaluation
- libs
	- classification (All scripts used for training and evaluation)
-  notebooks
- scripts (Executable scripts)
- models (Pretrained Models)


### Run

Download the Speech Commands Dataset and extract the dataset in the train folder. Test Audio can be placed in data/test/audio folder.

The notebooks can be run individually using Jupyter. To run the scripts from command line edit the notebooks using Jupyter and run:

    ./script/execute_notebook.py
   and select the notebook to run. The results are stored in results/notebook_name.log
   
   
P0 Predict Test WAV.ipynb can be used to predict audio files using a trained graphdef model.

This will open the Jupyter Notebook software and project file in your web browser.

### Training

The model was trained using a AWS instance with the following specifications:
- NVIDIA Tesla P100 X 1
- 16 GB RAM 
- 30 GB SSD

# Note
If there is any issue running the code, please post it in the issue tracker.
