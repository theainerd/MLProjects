# Predict Happy Customers

We will predict happy and unhappy customers which give companies a nice head-start to improve their experience.

## Data Description

`train.csv` contains 38932 rows along with the corresponding label (happy or not happy) and `test.csv` has 29404 rows.

## Model
I have Build models using [Catboost](https://github.com/Microsoft/LightGBM), [Lightgbm](https://github.com/catboost/catboost) and [NaiveBayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) algorithm in Python. Given the text classification problem, we will clean data, create bag of words matrix, tf-idf matrix.

## Running the model


Firstly, place the train.csv, test.csv in the input folder. Then, run the ipython notebook:

```console
theainerd:~$ jupyter notebook Customer Happiness.ipynb
```
## Dependencies

* numpy
* pandas
* nltk
* sklearn

# Note
If there is any issue running the code, please post it in the issue tracker.
