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
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']

    vectorizer = HashingVectorizer(n_features=2**10, binary=True, stop_words='english')
    x_train_vec = vectorizer.transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier_NB = BernoulliNB(alpha=1.0, binarize=None, fit_prior=True)
    classifier_NB.fit(X=x_train_vec, y=y_train)
    y_pred_NB = classifier_NB.predict(x_test_vec)
    accuracy_NB = accuracy_score(y_test, y_pred_NB)

    classifier_DT = DecisionTreeClassifier(max_depth=10)
    classifier_DT.fit(X=x_train_vec, y=y_train)
    y_pred_DT = classifier_DT.predict(x_test_vec)
    accuracy_DT = accuracy_score(y_test, y_pred_DT)

    print("Machine-learning algorithms - sklearn")
    print("Accuracy score:")
    print("\tNaive Bayes classifier:  ", accuracy_NB)
    print("\tDecision Tree classifier:", accuracy_DT)
    print("End of sklearn!")


def usingKeras() -> None:
    data = pickle.load(open("keras-data.pickle", "rb"))
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    vocab_size = data["vocab_size"]
    max_length = data["max_length"]

    LENGTH = 16
    x_train_prep = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=LENGTH)
    x_test_prep = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=LENGTH)
    y_train_prep = keras.utils.to_categorical(y_train, num_classes=2)
    y_test_prep = keras.utils.to_categorical(y_test, num_classes=2)

    DIM = 16
    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=DIM))
    model.add(keras.layers.LSTM(activation='sigmoid', units=DIM))
    model.add(keras.layers.Dense(units=2))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x=x_train_prep, y=y_train_prep, batch_size=DIM, epochs=3, verbose=1, use_multiprocessing=True)

    score = model.evaluate(x=x_test_prep, y=y_test_prep, verbose=1, use_multiprocessing=True)

    print("Deep learning - keras (TensorFlow)")
    print("Evaluation score:")
    print("\tLTSM loss:    ", score[0])
    print("\tLTSM accuracy:", score[1])
    print("End of keras!")


def main() -> None:
    print("TDT4171 - Exercise 5")

    # usingSklearn()
    # usingKeras()


if __name__ == '__main__':
    main()
