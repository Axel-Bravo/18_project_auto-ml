import pandas as pd
import numpy as np


class auto_clf (object):
    """
    Class that agglutinates all "auto" machine learning techniques for classification problems
    """

    def __init__(self, train, test):
        """
        Initializes the "auto_clf" object
        :param train: contains a columns named "Y"
        :type train: pd.DataFrame: all data needs to be numeric
        :param test: contains a column named "Y"
        type: test: pd.DataFrame: all data needs to be numeric
        """

        self._train = None
        self._test = None

    def load_from_file(self, path_train, path_test):
        """

        :param path_train:
        :param path_test:
        :return:
        """


    def classifiers_battle(self, **kwargs):
        """
        Runs a classifier
        :param kwargs:
        :return:
        """