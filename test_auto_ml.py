import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal

import AutoMl as ml
test_path = 'test/'


class TestLoad(unittest.TestCase):
    def setUp(self):
        data_train = pd.read_csv(test_path + 'load_from_file_ok' + '.csv')
        data_test = pd.read_csv(test_path + 'load_from_file_ok_2' + '.csv')
        path_train = test_path + 'load_from_file_ok_2' + '.csv'
        path_test = test_path + 'load_from_file_ok_2' + '.csv'
        self.data_test_pass = [(data_train, data_test), (path_train, path_test)]

        data_train_type = pd.read_csv(test_path + 'load_from_file_nok_col_type' + '.csv')
        path_train_type = test_path + 'load_from_file_nok_col_type' + '.csv'
        data_train_y = pd.read_csv(test_path + 'load_from_file_nok_y' + '.csv')
        path_train_y = test_path + 'load_from_file_nok_y' + '.csv'
        data_test_type = pd.read_csv(test_path + 'load_from_file_nok_col_type' + '.csv')
        path_test_type = test_path + 'load_from_file_nok_col_type' + '.csv'
        data_test_y = pd.read_csv(test_path + 'load_from_file_nok_y' + '.csv')
        path_test_y = test_path + 'load_from_file_nok_y' + '.csv'
        self.data_test_fail = [(data_train_type, data_test), (path_train_type, path_test), (data_train_y, data_test),
                               (path_train_y, path_test), (data_train, data_test_y), (path_train, path_test_y),
                               (data_train, data_test_type), (path_train, path_test_type)]

    def test_load_data_pass(self):
        for train, test in self.data_test_pass:
            clf_ml = ml.AutoML()
            clf_ml.load_data(train, test)

            if type(train) is str:
                assert_frame_equal(pd.read_csv(train), clf_ml._train)
                assert_frame_equal(pd.read_csv(test), clf_ml._test)
            else:
                assert_frame_equal(train, clf_ml._train)
                assert_frame_equal(test, clf_ml._test)

    def test_load_data_fail(self):
        for train, test in self.data_test_fail:
            clf_ml = ml.AutoML()
            self.assertRaises(Exception, clf_ml.load_data, train, test)


if __name__ == '__main__':
    unittest.main()
