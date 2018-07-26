import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal

import auto_ml as ml
test_path = 'test/'

class TestLoadFromFile(unittest.TestCase):

    def test_all_data_ok(self):
        """
        Function behaves properly on desired cases
        :return: AutoClf
            ._train: {pd.DataFrame}: train data
            ._test: {pd.DataFrame}: test data
        """
        data_train = pd.read_csv(test_path + 'load_from_file_ok' + '.csv')
        data_test = pd.read_csv(test_path + 'load_from_file_ok_2' + '.csv')
        train_data = test_path + 'load_from_file_ok' + '.csv'
        test_data = test_path + 'load_from_file_ok_2' + '.csv'

        clf_ml = ml.AutoClf()
        clf_ml.load(train_data=train_data, test_data=test_data)

        assert_frame_equal(data_train, clf_ml._train)
        assert_frame_equal(data_test, clf_ml._test)

    def test_column_types_train_data(self):
        """
        Check that only numerical data is accepted
        :return: None
        """
        train_data = test_path + 'load_from_file_nok_col_type' + '.csv'
        test_data = test_path + 'load_from_file_ok_2' + '.csv'

        clf_ml = ml.AutoClf()
        self.assertRaises(Exception, clf_ml.load, train_data, test_data)

    def test_column_types_test_data(self):
        """
        Check that only numerical data is accepted
        :return: None
        """
        train_data = test_path + 'load_from_file_ok_2' + '.csv'
        test_data = test_path + 'load_from_file_nok_col_type' + '.csv'

        clf_ml = ml.AutoClf()
        self.assertRaises(Exception, clf_ml.load, train_data, test_data)

    def test_column_names_train_data(self):
        """
        Check the column naming is appropriate, i.e. there is a column named "Y"
        :return: None
        """
        train_data = test_path + 'load_from_file_nok_y' + '.csv'
        test_data = test_path + 'load_from_file_ok_2' + '.csv'

        clf_ml = ml.AutoClf()
        self.assertRaises(Exception, clf_ml.load, train_data, test_data)

    def test_column_names_test_data(self):
        """
        Check the column naming is appropriate, i.e. there is a column named "Y"
        :return: None
        """
        train_data = test_path + 'load_from_file_ok_2' + '.csv'
        test_data = test_path + 'load_from_file_nok_y' + '.csv'

        clf_ml = ml.AutoClf()
        self.assertRaises(Exception, clf_ml.load, train_data, test_data)


class TestLoadFromMemory(unittest.TestCase):

    def test_all_data_ok(self):
        """
        Function behaves properly on desired cases
        :return: AutoClf
            ._train: {pd.DataFrame}: train data
            ._test: {pd.DataFrame}: test data
        """
        data_train = pd.read_csv(test_path + 'load_from_file_ok' + '.csv')
        data_test = pd.read_csv(test_path + 'load_from_file_ok_2' + '.csv')

        clf_ml = ml.AutoClf()
        clf_ml.load(train_data=data_train, test_data=data_test)

        assert_frame_equal(data_train, clf_ml._train)
        assert_frame_equal(data_test, clf_ml._test)

    def test_column_types_train_data(self):
        """
        Check that only numerical data is accepted
        :return: None
        """
        data_train = pd.read_csv(test_path + 'load_from_file_nok_col_type' + '.csv')
        data_test = pd.read_csv(test_path + 'load_from_file_ok_2' + '.csv')

        clf_ml = ml.AutoClf()
        self.assertRaises(Exception, clf_ml.load, data_train, data_test)

    def test_column_types_test_data(self):
        """
        Check that only numerical data is accepted
        :return: None
        """
        data_train = pd.read_csv(test_path + 'load_from_file_ok_2' + '.csv')
        data_test = pd.read_csv(test_path + 'load_from_file_nok_col_type' + '.csv')

        clf_ml = ml.AutoClf()
        self.assertRaises(Exception, clf_ml.load, data_train, data_test)

    def test_column_names_train_data(self):
        """
        Check the column naming is appropriate, i.e. there is a column named "Y"
        :return: None
        """
        data_train = pd.read_csv(test_path + 'load_from_file_nok_y' + '.csv')
        data_test = pd.read_csv(test_path + 'load_from_file_ok_2' + '.csv')

        clf_ml = ml.AutoClf()
        self.assertRaises(Exception, clf_ml.load, data_train, data_test)

    def test_column_names_test_data(self):
        """
        Check the column naming is appropriate, i.e. there is a column named "Y"
        :return: None
        """
        data_train = pd.read_csv(test_path + 'load_from_file_ok_2' + '.csv')
        data_test = pd.read_csv(test_path + 'load_from_file_nok_y' + '.csv')

        clf_ml = ml.AutoClf()
        self.assertRaises(Exception, clf_ml.load, data_train, data_test)


if __name__ == '__main__':
    unittest.main()
