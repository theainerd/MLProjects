# Predict Happy Customers

We will predict happy and unhappy customers which give companies a nice head-start to improve their experience.


### Install

This project requires **Python3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [nltk](https://www.nltk.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included

### Code

Code is provided in the notebook `Customer Happiness.ipynb` notebook file. Train and test dataset can be found in `input` folder. If you want to ensemble different models then you can find the code to do that in `Ensembling codes` folder. 

### Run

In a terminal or command window, navigate to the top-level project directory `` (that contains this README) and run one of the following commands:

```console
theainerd:~$ jupyter notebook Customer Happiness.ipynb
```
or
```bash
theainerd:~$ ipython notebook  Customer Happiness.ipynb
```

This will open the Jupyter Notebook software and project file in your web browser.

## Model
I have Build models using [Catboost](https://github.com/Microsoft/LightGBM), [Lightgbm](https://github.com/catboost/catboost) and [NaiveBayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) algorithm in Python. Given the text classification problem, we will clean data, create bag of words matrix, tf-idf matrix.

### Data

The dataset used in this project is included as `train.csv`. This dataset is provided by TripAdvisor and contains the following attributes:

**Features**
- `User_ID` :  unique ID of the customer
- `Description` : description of the review posted
- `Browser_Used` : browser used to post the review
- `Device_Used`: device used to post the review 


**Target Variable**
- `Happy or Not Happy` :  Is_Response (0 = No; 1 = Yes)

### Note
If there is any issue running the code, please post it in the issue tracker.
