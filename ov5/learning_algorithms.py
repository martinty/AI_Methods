from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import pickle

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow import keras


def usingSklearn() -> None:
    data = pickle.load(open("sklearn-data.pickle", "rb"))

    print("End of sklearn!")


def usingKeras() -> None:
    data = pickle.load(open("keras-data.pickle", "rb"))
    model = keras.Sequential()

    # keras.preprocessing.sequence.pad_sequences()
    # model.add(keras.layers.Embedding())
    # model.add(keras.layers.LSTM())
    # model.add(keras.layers.Dense())
    # model.fit()
    # model.evaluate()

    print("End of keras!")


def main() -> None:
    print("TDT4171 - Exercise 5")

    usingSklearn()
    usingKeras()


if __name__ == '__main__':
    main()
