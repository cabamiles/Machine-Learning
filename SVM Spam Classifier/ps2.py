from __future__ import division
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import time
from math import log


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


class SVM:
    """
    Implementation of the Support Vector Machine using the Pegasos algorithm
    """

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.w = np.array([])
        self.obj_list = []
        self.avg_hinge_loss = 0
        self.support_vectors = 0

    def pegasos_svm_train(self, x, y, lamb, num_iter=20):
        self.support_vectors = 0
        # Check if inputs are valid
        assert(type(x) == np.ndarray and type(y) == np.ndarray)
        assert(x.shape[0] == y.shape[0])

        # Set our data fields
        self.x = x
        self.y = y

        # Useful variables to store
        num_features = self.x.shape[1]
        num_examples = self.x.shape[0]

        # initialize weight vector and average weight vector
        w = np.zeros((num_features,), dtype='int64')

        # store the value of the SVM objective function per each pass through the data
        objective = []
        # store the sum of the hinge loss
        hinge = 0
        # store the sum of the training error
        t = 0
        for _ in xrange(num_iter):
            # count the number of mistakes made on this iteration
            obj_sum = 0
            for j in range(num_examples):
                # classifying every email in our training data
                t += 1
                eta_t = 1/(lamb*t)
                x_j = x[j]
                y_j = 1
                if y[j] == 0:
                    y_j = -1

                # y_i will be our predicted value of the example
                obj_sum += max(0, 1-(y_j*np.dot(w, x_j)))
                if y_j*np.dot(x_j, w) < 1:
                    w = ((1-eta_t*lamb)*w) + (eta_t*y_j*x_j)
                else:
                    w = (1-eta_t*lamb)*w
            # increment total number of mistakes by the number of mistakes made on this pass of the data
            objective.append(((lamb/2)*(np.dot(w, w))) + ((obj_sum+max(0, 1-(y[num_examples-1] *
                                                                    np.dot(w, x[num_examples-1]))))/num_examples))
            hinge = obj_sum/num_examples
        self.w = w
        self.obj_list = objective
        self.avg_hinge_loss = hinge
        # return the weight vector, total number of mistakes made, and the number of passes through the data
        return self.w

    def pegasos_svm_test(self, x_test, y_test, w=None):

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
            y_t = y_test[t] if y_test[t] == 1 else -1
            r = np.dot(w, x_t)
            if r <= 1 and r >= -1:
                self.support_vectors += 1
            # predict spam if the feature vector's dot product with the weight vector is greater than or equal to zero
            if y_t*r < 0:
                num_error += 1


        # return Decimal fraction for accurate precision
        return num_error/num_examples

print "Processing data..."
x, y, x_val, y_val = process_data('spam_train.txt')
num_examples = x.shape[0]
# initialize a BinaryVectorizor object
vectorizor = BinaryVectorizor()
# create the vocabulary that is going to be used with the training data and vectorize the training data
x = vectorizor.fit_transform(x)

# initialize a Perceptron object
clf = SVM()
print "Training SVM..."
start = time.time()
clf.pegasos_svm_train(x, y, 2**(-5))
print "Training time: %f" % (time.time() - start)
num_iter = range(20)
num_iter = [(i+1) for i in num_iter]
# print clf.obj_list
plt.plot(num_iter, clf.obj_list)
plt.show()

x_val = vectorizor.transform(x_val)
lambda_list = range(-9, 2)
lambda_list = [2**i for i in lambda_list]
log_lambda = [log(i) for i in lambda_list]
hinge_list = []
train_err_list = []
val_err_list = []
num_support_vectors = []
print "Training on multiple regularization parameters..."
for l in lambda_list:
    clf.pegasos_svm_train(x, y, l)
    hinge_list.append(clf.avg_hinge_loss)
    train_err_list.append(clf.pegasos_svm_test(x, y))
    num_support_vectors.append(clf.support_vectors)
    val_err = clf.pegasos_svm_test(x_val, y_val)
    val_err_list.append(val_err)

plt.plot(log_lambda,
         train_err_list,
         'r--',
         log_lambda,
         hinge_list,
         'g--')
plt.show()

print "Minimum validation error: %f" % min(val_err_list)

optimal_lambda = lambda_list[val_err_list.index(min(val_err_list))]
clf.pegasos_svm_train(x, y, optimal_lambda)

x_test, y_test = process_data('spam_test.txt', validation=False)
x_test = vectorizor.transform(x_test)

test_err = clf.pegasos_svm_test(x_test, y_test)
print "Number of support vectors for optimal lambda: %f" % num_support_vectors[val_err_list.index(min(val_err_list))]
print "Minimum test error: %f" % test_err
