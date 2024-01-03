"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List
import numpy as np
import scipy.linalg

N_DIMENSIONS = 10


def classify(pcatrain_data: np.ndarray, labels_training: np.ndarray, pcatest_data: np.ndarray, v, test_mean, train_mean) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # getting back the full data

    v = v[:, 2:]
    pcatrain_data = pcatrain_data[:, 2:]
    pcatest_data = pcatest_data[:, 2:]
    training_data = np.dot(pcatrain_data, v.transpose()) + train_mean
    testing_data = np.dot(pcatest_data, v.transpose()) + test_mean

    
    #Performing the nearest neighbour classifier using the reduced feature vectors from the model

    x = np.dot(testing_data, training_data.transpose())
    mod_testing_data = np.sqrt(np.sum(testing_data * testing_data, axis=1))
    mod_training_data = np.sqrt(np.sum(training_data * training_data, axis=1))
    dist = x / np.outer(mod_testing_data, mod_training_data.transpose())  # cosine distance

    nearest = np.argmax(dist, axis=1) # returns array of the indicies of the nearest values
    labels = labels_training[nearest]

    return labels



# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(feature_data: np.ndarray, model: dict = None) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    mean = np.mean(feature_data, axis=0)

    covx = np.cov(feature_data, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, subset_by_index=(N - N_DIMENSIONS, N - 1)) #eigen values, eigen vectors
    v = np.fliplr(v)

    print("num of dimensions is:", w.shape[0])

    pca_data = np.dot((feature_data - np.mean(feature_data)), v)

    
    if model == {}:
        model['train_mean'] = mean.tolist()
    else:
        model['test_mean'] = mean.tolist()
    model['v'] = v.tolist()
    
    return pca_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["labels_train"] = labels_train.tolist()
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors

    


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])
    v = np.array(model["v"])
    test_mean = np.array(model["test_mean"])
    train_mean = np.array(model["train_mean"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test, v, test_mean, train_mean)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)
