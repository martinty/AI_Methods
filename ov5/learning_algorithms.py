import time
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import pickle
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from tensorflow import keras


def verify_reviews(y):
    good = 0
    for i in y:
        good += i
    score = good/len(y)
    print("Good reviews in test set:", score)
    print("Bad reviews in test set:", 1-score, "\n" + "-" * 100)


def usingSklearn() -> None:
    data = pickle.load(open("sklearn-data.pickle", "rb"))
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']

    vectorizer = HashingVectorizer(n_features=2**10, binary=True, stop_words='english')
    x_train_vec = vectorizer.transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    #  Naive Bayes classifier
    start_time_NB = time.time()
    classifier_NB = BernoulliNB(alpha=1.0, binarize=None, fit_prior=True)
    classifier_NB.fit(X=x_train_vec, y=y_train)
    y_pred_NB = classifier_NB.predict(x_test_vec)
    accuracy_NB = accuracy_score(y_test, y_pred_NB)
    elapsed_time_NB = time.time() - start_time_NB

    # Decision Tree classifier
    start_time_DT = time.time()
    classifier_DT = DecisionTreeClassifier(max_depth=10)
    classifier_DT.fit(X=x_train_vec, y=y_train)
    y_pred_DT = classifier_DT.predict(x_test_vec)
    accuracy_DT = accuracy_score(y_test, y_pred_DT)
    elapsed_time_DT = time.time() - start_time_DT

    print("-" * 100, "\nMachine-learning algorithms - sklearn")
    verify_reviews(y_test)
    print("Accuracy score:")
    print("\tNaive Bayes classifier:  ", accuracy_NB)
    print("\tDecision Tree classifier:", accuracy_DT, "\n" + "-" * 100)
    print("Elapsed time in seconds:")
    print("\tNaive Bayes classifier:  ", elapsed_time_NB)
    print("\tDecision Tree classifier:", elapsed_time_DT, "\n" + "-" * 100)


def usingKeras() -> None:
    data = pickle.load(open("keras-data.pickle", "rb"))
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    vocab_size = data["vocab_size"]
    max_length = data["max_length"]

    LENGTH = max_length
    x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=LENGTH)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=LENGTH)
    y_train_binary = keras.utils.to_categorical(y_train)
    y_test_binary = keras.utils.to_categorical(y_test)

    DIM = 128
    load = False
    save = True

    model = keras.Sequential()
    if load:
        model = keras.models.load_model('my_model.h5')
    else:
        model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=DIM, input_length=LENGTH))
        model.add(keras.layers.LSTM(activation='sigmoid', units=DIM))
        model.add(keras.layers.Dense(units=2))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x=x_train_pad, y=y_train_binary, batch_size=DIM, epochs=6, verbose=1, use_multiprocessing=True)
        if save:
            model.save('my_model.h5')

    print("-" * 100, "\nDeep learning - keras (TensorFlow)")
    verify_reviews(y_test)
    print("Evaluation score:")
    loss, accuracy = model.evaluate(x=x_test_pad, y=y_test_binary, verbose=1, use_multiprocessing=True)
    print("\tLTSM loss:    ", loss)
    print("\tLTSM accuracy:", accuracy, "\n" + "-" * 100)
    print(model.summary(), "\n" + "-" * 100)


def main() -> None:
    print("TDT4171 - Exercise 5")

    usingSklearn()
    usingKeras()


if __name__ == '__main__':
    main()
