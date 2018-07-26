import pandas as pd
import numpy as np


class AutoClf (object):
    """
    Class that agglutinates all "auto" machine learning techniques for classification problems
    """

    def __init__(self):
        """
        Initializes the "auto_clf" object

        """
        self._train = None
        self._test = None

    def load_from_file(self, path_train, path_test):
        """
        Load data from a ".csv" fle
        :param path_train: file path to train data
        :type path_train: str
        :param path_test: file path to test data
        :type path_test: str
        :return: Initialize object
        """
        names = ['Train', 'Test']
        paths = [path_train, path_test]

        for name, path in zip(names, paths):
            temp = pd.read_csv(path)
            try:
                assert_alert = name + ' data column types are not numeric'
                assert(temp.shape == temp._get_numeric_data().shape), assert_alert

                assert_alert = name + ' data has no "Y" column'
                assert ("Y" in temp.columns), assert_alert

                if name is "Train":
                    self._train = temp
                else:
                    self._test = temp

            except AssertionError:
                raise Exception(name + ' data does not follow requirements')

    def load_from_memory(self, train_object, test_object):
        """
        Load data from the memory
        :param train_object: train data
        :type pd.DataFrame
        :param test_object: test data
        :type pd.DataFrame
        :return: Initialize object
        """

        names = ['Train', 'Test']
        instances = [train_object, test_object]

        for name, instance in zip(names, instances):

            try:
                assert_alert = name + ' data column types are not numeric'
                assert(instance.shape == instance._get_numeric_data().shape), assert_alert

                assert_alert = name + ' data has no "Y" column'
                assert ("Y" in instance.columns), assert_alert

                if name is "Train":
                    self._train = instance
                else:
                    self._test = instance

            except AssertionError:
                raise Exception(name + ' data does not follow requirements')


    def classifiers_battle(self, **kwargs):
        """
        Runs a classifier
        :param kwargs:
        :return:
        """
        # TODO: pending
        pass