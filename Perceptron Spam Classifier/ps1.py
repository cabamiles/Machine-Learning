import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from decimal import *
import numpy as np
import sys


def process_data(file, validation=True):
    """
    Preprocessing of input data. Partitions training data and training labels
    as well as validation data and labels.
    :param file: .txt file with an int 1 or 0 as the first character of each line
                 and email preprocessed email text as the rest of the line
    :return: x_train, y_train, x_val, y_ val as numpy arrays
    """
    if validation:
        x_train = []
        y_train = []
        db = open(file, 'r')
        for line in db:
            # add the label to y and the corresponding text to x
            y_train.append(line[0])
            x_train.append(line[2:])

        x_val = x_train[4000:]
        del x_train[4000:]
        y_val = y_train[4000:]
        del y_train[4000:]
        x_train = np.array(x_train, dtype='str')
        y_train = np.ravel(np.array(y_train, dtype='int64'))
        x_val = np.array(x_val, dtype='str')
        y_val = np.ravel(np.array(y_val, dtype='int64'))

        return x_train, y_train, x_val, y_val

    else:
        x = []
        y = []
        db = open(file, 'r')
        for line in db:
            # add the label to y and the corresponding text to x
            y.append(line[0])
            x.append(line[2:])
        x = np.array(x, dtype='str')
        y = np.ravel(np.array(y, dtype='int64'))

        return x, y

class BinaryVectorizor:
    """
    This class represents a Bag of Words and provides methods for creating a vocabulary and vectorization.

    Data Fields
    _______________________________________________________
    vocab: list containing the bag of all words that occur in at least 30 documents
    vocab_size: int storing the size of the vocab
    """
    def __init__(self):
        self.vocab = []
        self.vocab_size = 0

    def fit_transform(self, data):
        """
        Create the vocabulary given a corpus of documents and transform the given data into binary vector representation
        :param data: array-like corpus of text to create the vocabulary
        :return: new_data ndarray vectorized representation of data
        """
        # initialize vocabulary dictionary to determine the number of emails in which each distinct word occurs
        vocab_dict = {}
        # initialize vectorized data set
        new_data = []
        for line in data:
            # for every document in the data set create determine all unique words in that document
            set_line = set(line.split(' '))
            # for every unique word in that document if that word is already in the vocab_dict, increment the number
            # of documents in which it occurs
            # otherwise, add it to the vocab_dict
            for word in set_line:
                if word in vocab_dict:
                    vocab_dict[word] = (vocab_dict[word] + 1)
                else:
                    vocab_dict[word] = 1

        # for every word in the vocab_dict, if the word appears less in less than 30 email, disreagard it
        # otherwise append it to objects vocabulary
        for key, val in vocab_dict.items():
            if val >= 30:
                self.vocab.append(key)
        # sort the objects vocabulary to get deterministic representation of vectors
        self.vocab.sort()
        self.vocab_size = len(self.vocab)

        for line in data:
            # vectorize every line in document given the new vocab
            vect = self.vectorize(line)
            # append the vectorized representation of the document to the new data
            new_data.append(vect)

        # return array representation of the new data
        return np.array(new_data)

    def transform(self, data):
        """
        Transform a data set of document strings into a dataset of binary feature vectors using the object's vocabulary
        :param data: array-like dataset containing document strings
        :return: ndarray dataset of binary feature vectors
        """
        new_data = []
        for line in data:
            vect = self.vectorize(line)
            new_data.append(vect)

        return np.array(new_data)

    def get_vocab(self):
        # method that returns the objects vocabulary
        return self.vocab

    def vectorize(self, data):
        """
        Vectorize a document string into a binary feature vector
        :param data: String document string
        :return: ndarray of vectorized binary feature vector
        """
        data = data.split(' ')
        bin_vect = np.zeros((1, len(self.vocab)))
        for word in data:
            if word in self.vocab:
                idx = self.vocab.index(word)
                bin_vect[0, idx] = 1
        return np.ravel(bin_vect).tolist()

class Perceptron:
    """
    This class represents a perceptron algorithm. It provides methods for training the algorithm with different
    parameters, testing the trained model, and predicting a binary class given a feature vector

    Data fields
    ___________________________
    x: ndarray containing feature vectors that is used for training
    y: ndarray containing corresponding labels used for training
    w: trained weight vector
    """
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.w = np.array([])

    def perceptron_train(self, x, y, average=False, max_iter=None):
        """
        Train the perceptron algorithm
        :param x: ndarray with N features and M examples used for training
        :param y: 1D array with corresponding M labels
        :param average: (optional) Default: False. If True, performs the Average Perceptron algorithm
        :param max_iter: (optional) Default: None. If an int is given, only performs that number
                                            of passes through the data
        :return: w, k, num_iter: ndarray learned weight vector, int total number of mistakes made,
                                and int total number of passes through the data respectively
        """

        # Check if inputs are valid
        assert(type(x) == np.ndarray and type(y) == np.ndarray)
        assert(x.shape[0] == y.shape[0])

        # Set our data fields
        self.x = x
        self.y = y
        # If we are using the default max_iter, we set our max_iter to the largest value of an integer
        if (max_iter is None):
            max_iter = sys.maxint
        # Useful variables to store
        num_features = self.x.shape[1]
        num_examples = self.x.shape[0]

        # initialize weight vector and average weight vector
        w = np.zeros((num_features,), dtype='int64')
        w_avg = w
        # sentinel variable for stopping the learning process
        all_correct = False
        # count the number of passes through the data
        num_iter = 0
        # count the total number of mistakes made
        k = 0

        while (not all_correct) and (num_iter < max_iter):
            # count the number of mistakes made on this iteration
            num_mistakes = 0
            for t in range(num_examples):
                # classifying every email in our training data
                x_t = x[t]
                y_t = 1
                if y[t] == 0:
                    y_t = -1
                # y_i will be our predicted value of the example
                y_i = 0
                if np.dot(w, x_t) >= 0:
                    # if the dot product is >= predict as spam
                    y_i = 1
                elif np.dot(w, x_t) < 0:
                    # otherwise predict as not spam
                    y_i = -1
                # if our predicted label is not correct, update the weight vector
                if y_i != y_t:
                    num_mistakes+=1
                    w = np.add(w, y_t*x_t)
                # add attempted weight vector to the average weight vector
                w_avg = np.add(w_avg, w)
            # increment total number of mistakes by the number of mistakes made on this pass of the data
            k+=num_mistakes
            num_iter+=1
            # if no mistakes were made on this iteration, stop the loop
            if num_mistakes == 0:
                all_correct = True
        # divide the average weight vector by the total number of weight vectors considered
        w_avg = np.divide(w_avg, num_iter*num_examples)
        # if using the Average Perceptron algorithm, set the weight objects weight vector as the average weight vector
        if average:
            self.w = w_avg
        # otherwise set it as the unaveraged trained weight vector
        else:
            self.w = w
        # return the weight vector, total number of mistakes made, and the number of passes through the data
        return self.w, k, num_iter

    def perceptron_test(self, x_test, y_test, w=None):
        """
        Produce the test error
        :param x_test: ndarray of vectorized test data with N features and M examples
        :param y_test: ndarray of corresponding M labels
        :param w: (optional) ndarray trained weight vector. Uses object's weight vector if None
        :return: Decimal total number of errors made on test set divided by total number of examples
        """
        if w is None:
            w = self.w

        # validate input data
        assert(type(x_test) == np.ndarray and type(y_test) == np.ndarray and type(w) == np.ndarray)
        assert(w.shape[0] == x_test.shape[1])
        assert(x_test.shape[0] == y_test.shape[0])

        # store useful variables
        num_examples = x_test.shape[0]
        num_error = 0
        # make a prediction for every example in test data
        for t in range(num_examples):
            x_t = x_test[t]
            y_t = y_test[t]
            # initialize our predicted class as not spam
            y_i = 0
            prediction = np.dot(w, x_t)
            # predict spam if the feature vector's dot product with the weight vector is greater than or equal to zero
            if prediction >= 0:
                y_i = 1
            # if our prediction is wrong, increment the number of errors that we made
            if y_i != y_t:
                num_error+=1

        # return Decimal fraction for accurate precision
        return Decimal(num_error)/Decimal(num_examples)

    def predict(self, x, w=None):
        """
        Predict the class of a single email
        :param x: ndarray of single vectorized example
        :param w: (optional) ndarray of weight vector
        :return: int predicted class of the example
        """
        if w is None:
            w = self.w

        assert(type(x) == np.ndarray and type(w) == np.ndarray)
        assert(w.shape[0] == x.shape[0])

        y_i = 0
        prediction = np.dot(w, x)
        if prediction >= 0:
            y_i = 1
        return y_i

# process the spam training set
x, y, x_val, y_val = process_data('spam_train.txt')
# initialize a BinaryVectorizor object
vectorizor = BinaryVectorizor()
# create the vocabulary that is going to be used with the training data and vectorize the training data
x = vectorizor.fit_transform(x)

# initialize a Perceptron object
clf = Perceptron()

# train the Perceptron Algorithm with the training data
w, k, num_iter = clf.perceptron_train(x, y)

# test that the Perceptron Algorithm returns a zero training error
err_train = Decimal(clf.perceptron_test(x, y))

print "Total number of mistakes made before convergence: %d" % k
print "Number of passes through data: %d" % num_iter
print "Training error: %f" % err_train

# vectorize the validation data
x_val = vectorizor.transform(x_val)
# compute the validation error
err_val = Decimal(clf.perceptron_test(x_val, y_val))
print "Validation error: %f" % err_val

# using the vocabulary and the trained weight vector
# print the most positive and the most negative words and their weights
vocab_dict = {}
vocab = vectorizor.get_vocab()

for i in range(len(vocab)):
    vocab_dict[vocab[i]] = w[i]

sorted_dict = sorted(vocab_dict.items(), key=lambda t: t[1])
least = sorted_dict[0:15]
most = sorted_dict[len(sorted_dict)-15:]
most.reverse()

print most
print least

# plot the number validation error as a function of the number of examples used to train
# for both the Perceptron Algorithm and the Average Perceptron Algorithm
# also plot number of passes over the data as a function of the number of examples used
N = [100, 200, 400, 800, 2000, 4000]
val_err = []
val_err_avg = []
iter_list = []
iter_list_avg = []
for n in N:
    w, k, num_iter = clf.perceptron_train(x[:n], y[:n])
    w_avg, k_avg, num_iter_avg = clf.perceptron_train(x[:n], y[:n], average=True)
    iter_list.append(num_iter)
    iter_list_avg.append(num_iter_avg)
    val_err.append(clf.perceptron_test(x_val, y_val, w))
    val_err_avg.append(clf.perceptron_test(x_val, y_val, w_avg))

plt.plot(N, val_err, 'r--', N, val_err_avg, 'b--')
plt.show()

plt.plot(N, iter_list, 'r--', N, iter_list_avg, 'bs')
plt.show()

# train the model with different parameters with the training data and test against the validation data
# to try to determine best parameters for testing
err_list = []
weight_list = []
# Train perceptron algorithm with no limit on max_iter
w0 = clf.perceptron_train(x, y)[0]
err_list.append(clf.perceptron_test(x_val, y_val, w0))
weight_list.append(w0)
# Train perceptron average algorithm with no limit on max_iter
w1 = clf.perceptron_train(x, y, average=True)[0]
err_list.append(clf.perceptron_test(x_val, y_val, w1))
weight_list.append(w1)
# Train perceptron algorithm with max_iter as 5
w2 = clf.perceptron_train(x, y, max_iter=5)[0]
err_list.append(clf.perceptron_test(x_val, y_val, w2))
weight_list.append(w2)
# Train perceptron average algorithm with max_iter as 5
w3 = clf.perceptron_train(x, y, average=True, max_iter=5)[0]
err_list.append(clf.perceptron_test(x_val, y_val, w3))
weight_list.append(w3)
# Train perceptron algorithm with max_iter as 10
w4 = clf.perceptron_train(x, y, max_iter=10)[0]
err_list.append(clf.perceptron_test(x_val, y_val, w4))
weight_list.append(w4)
# Train perceptron average algorithm with max_iter as 10
w5 = clf.perceptron_train(x, y, average=True, max_iter=10)[0]
err_list.append(clf.perceptron_test(x_val, y_val, w5))
weight_list.append(w5)


# Finally, test the optimal model on the test data
# use the complete training data for training
x, y = process_data('spam_train.txt', validation=False)
x = vectorizor.transform(x)

# train the model with optimal parameters
optimal_index = err_list.index(min(err_list))
print err_list
print optimal_index
if optimal_index == 0:
    clf.perceptron_train(x, y)
elif optimal_index == 1:
    clf.perceptron_train(x, y, average=True)
elif optimal_index == 2:
    clf.perceptron_train(x, y, max_iter=5)
elif optimal_index == 3:
    clf.perceptron_train(x, y, average=True, max_iter=5)
elif optimal_index == 4:
    clf.perceptron_train(x, y, max_iter=10)
else:
    clf.perceptron_train(x, y, average=True, max_iter=10)


# process the training data
x_test, y_test = process_data('spam_test.txt', validation=False)
# vectorize the training data
x_test = vectorizor.transform(x_test)

# compute the test error and test accuracy of the model
test_error = Decimal(clf.perceptron_test(x_test, y_test))
test_accuracy = (1 - test_error) * 100
print "Test accuracy: %.2f%%" % test_accuracy
print "Test error: %f" % test_error