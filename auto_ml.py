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

    def load(self, train_data, test_data):
        """
        Load data from a ".csv" file or from memory
        :param train_data: file path to train data
        :type train_data: str
        :param test_data: file path to test data
        :type test_data: str
        :return: Initialize object
        """
        names = ['Train', 'Test']
        datas = [train_data, test_data]

        for name, data in zip(names, datas):
            if type(data) is str:
                temp = pd.read_csv(data)
                print('Loading {} from csv file'.format(name))
            else:
                temp = data
                print('Loading {} from memory'.format(name))

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

    def model_battle(self, **kwargs):
        """
        Runs a classifier
        :param kwargs:
        :return:
        """
        # TODO: pending
        pass