import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer
import time
import matplotlib.pyplot as plt
import scikitplot as skplt
import tensorflow as tf
from tensorflow import keras


def verify_reviews(y) -> None:
    # Print % of good and bad reviews
    good = 0
    for i in y:
        good += i
    score = good/len(y)
    print("Good reviews in test set:", score)
    print("Bad reviews in test set:", 1-score, "\n" + "-" * 100)


def usingSklearn() -> None:
    # Load data from data set
    data = pickle.load(open("sklearn-data.pickle", "rb"))
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # Adjust data
    vectorizer = HashingVectorizer(n_features=2**14, binary=True, stop_words='english')
    x_train_vec = vectorizer.transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # Naive Bayes classifier
    start_time_NB = time.time()
    classifier_NB = BernoulliNB(alpha=1.0, binarize=None)
    classifier_NB.fit(X=x_train_vec, y=y_train)
    y_pred_NB = classifier_NB.predict(x_test_vec)
    accuracy_NB = accuracy_score(y_test, y_pred_NB)
    elapsed_time_NB = time.time() - start_time_NB

    # Decision Tree classifier
    start_time_DT = time.time()
    classifier_DT = DecisionTreeClassifier(max_depth=8)
    classifier_DT.fit(X=x_train_vec, y=y_train)
    y_pred_DT = classifier_DT.predict(x_test_vec)
    accuracy_DT = accuracy_score(y_test, y_pred_DT)
    elapsed_time_DT = time.time() - start_time_DT

    plot = False
    if plot:
        # Plot Confusion Matrix
        skplt.metrics.plot_confusion_matrix(y_test, y_pred_NB, normalize=True,
                                            title="Normalized Confusion Matrix for BernoulliNB")
        plt.savefig('confusion_matrix_NB.pdf')
        plt.show()
        skplt.metrics.plot_confusion_matrix(y_test, y_pred_DT, normalize=True,
                                            title="Normalized Confusion Matrix for DecisionTreeClassifier")
        plt.savefig('confusion_matrix_DT.pdf')
        plt.show()

    # Result
    print("-" * 100, "\nMachine-learning algorithms - sklearn")
    verify_reviews(y_test)
    print("Accuracy score:")
    print("\tNaive Bayes classifier:  ", accuracy_NB)
    print("\tDecision Tree classifier:", accuracy_DT, "\n" + "-" * 100)
    print("Elapsed time in seconds:")
    print("\tNaive Bayes classifier:  ", elapsed_time_NB)
    print("\tDecision Tree classifier:", elapsed_time_DT, "\n" + "-" * 100)


def usingKeras() -> None:
    # Load data from data set
    data = pickle.load(open("keras-data.pickle", "rb"))
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    vocab_size = data["vocab_size"]
    max_length = data["max_length"]

    # Adjust data set
    x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=int(max_length * 1/2))
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=int(max_length * 1/2))
    y_train_binary = keras.utils.to_categorical(y_train)
    y_test_binary = keras.utils.to_categorical(y_test)

    # Options (model)
    load = False
    save = True
    plot = True

    # Build and train model
    model = keras.Sequential()
    if load:
        model = keras.models.load_model('my_model.h5')
    else:
        model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=2))
        model.add(keras.layers.LSTM(units=2))
        model.add(keras.layers.Dense(units=2))
        model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
        history = model.fit(x=x_train_pad, y=y_train_binary, validation_data=[x_test_pad, y_test_binary],
                            epochs=3, batch_size=2**10, verbose=2)
        if save:
            model.save('my_model.h5')
        if plot:
            # Plot training & validation accuracy values
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig('accuracy_keras.pdf')
            plt.show()
            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig('loss_keras.pdf')
            plt.show()

    if plot:
        # Plot Confusion Matrix
        y_pred = model.predict_classes(x=x_test_pad, batch_size=2**10, verbose=2)
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True,
                                            title="Normalized Confusion Matrix for LSTM")
        plt.savefig('confusion_matrix_LSTM.pdf')
        plt.show()

    # Result
    print("-" * 100, "\nDeep learning - keras (TensorFlow)")
    verify_reviews(y_test)
    print("Evaluation score:")
    loss, accuracy = model.evaluate(x=x_test_pad, y=y_test_binary, batch_size=2**10, verbose=2)
    print("\tLTSM loss:    ", loss)
    print("\tLTSM accuracy:", accuracy, "\n" + "-" * 100)
    print(model.summary(), "\n" + "-" * 100)


def main() -> None:
    print("TDT4171 - Exercise 5")

    # usingSklearn()
    usingKeras()


if __name__ == '__main__':
    main()
