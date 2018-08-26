import numpy as np
import pandas as pd
from scipy.stats import randint, expon, uniform
from random import choices

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib


class AutoMl (object):
    """
    Class that agglutinates all "auto" machine learning techniques for classification problems

    Modes available:
        - Supervised | Classification
        - Supervised | Regression

    Modality:
        - 'kaggle': mode suited for Kaggle competitions, i.e., we do not have the "y_test"
        - 'regular': mode suited for cases where we have complete train and test sets
    """

    __slots__ = ['_name', '_x_train', '_x_test', '_y_train', '_y_test', '_category', '_goal', '_modality', '_models']

    def __init__(self, name: str, category: str = 'supervised', goal: str = 'classification', modality: str = 'kaggle'):

        self._name: str = name

        # Data for the model
        self._x_train: pd.DataFrame = None
        self._y_train: pd.DataFrame = None
        self._x_test: pd.DataFrame = None
        self._y_test: pd.DataFrame = None

        # Type of model to build
        self._category: str = category
        self._goal: str = goal
        self._modality: str = modality

        # Results of the model training
        self._models: dict = None

    # I/O methods
    def load_data(self, train_data, test_data):
        """
        Load data from a ".csv" file or from memory
        :param train_data: file path to train data
        :type train_data: str
        :param test_data: file path to test data
        :type test_data: str
        :return: Initialize object
        """

        if self._category is 'supervised':
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

                    if self._modality is 'kaggle':
                        if name is "Train":
                            assert_alert = name + ' data has no "Y" column'
                            assert ("Y" in temp.columns), assert_alert
                            self._y_train = temp['Y']
                            self._x_train = temp.drop(columns=['Y'])
                        else:
                            self._y_test = temp['Y']
                            self._x_test = temp.drop(columns=['Y'])

                    if self._modality is 'regular':
                        assert_alert = name + ' data has no "Y" column'
                        assert ("Y" in temp.columns), assert_alert

                        if name is "Train":
                            self._y_train = temp['Y']
                            self._x_train = temp.drop(columns=['Y'])
                        else:
                            self._x_test = temp.drop(columns=['Y'])

                except AssertionError:
                    raise Exception(name + ' data does not follow requirements')

    def save_model(self, file_path: str, best_model: bool = False):
        """
        Saves the model into de disk on a specified path
        :param file_path: file path on the disk
        :param best_model: True: save only the best model | False: save the best model fo each type of algorithm
        :return: None | saves model into disk
        """
        if best_model:
            joblib.dump(self._models[0][1][1].best_estimator_, filename=file_path + self._name + '.pkl', compress=3)
        else:
            joblib.dump(self._models, filename=file_path + self._name + '.pkl', compress=3)

    # Executing methods
    def baseline_algorithm(self):
        from sklearn.metrics import accuracy_score, mean_squared_error

        """
        Give the results of the algorithm if we use a naive approach, i.e. use the mean value for regression and the
        :return:
        """
        if self._modality is 'kaggle':
            print('Kaggle modality does no allow to check the algorithm tru performance on local')

        if self._modality is 'regular':
            if self._category is 'supervised' and self._goal is 'classification':
                # We generate a naive prediction sample based on the distribution of the "y_train" sample
                freq_table = self._y_train.groupby(self._y_train).size()
                population = list(freq_table.index)
                weights = list(freq_table.values/sum(freq_table.values))
                y_baseline_pred = choices(population, weights, k=len(self._y_test))
                print('The naive accuracy score is: {}'.format(accuracy_score(self._y_test, y_baseline_pred)))

            if self._category is 'supervised' and self._goal is 'regression':
                # We generate a naive prediction sample based on the mean value of the "y_train" sample
                y_baseline_pred = [self._y_train.mean()] * len(self._y_test)
                print('The naive accuracy score is: {}'.format(mean_squared_error(self._y_test, y_baseline_pred)))

    def optimize(self, n_jobs: int = 2):
        """
        Realizes the battle model optimization
        :return:
        """

        models = {}
        parameters = self._load_parameters()
        pca_params = {'pca__n_components': randint(2, self._x_train.shape[1])}
        evaluator = self._evaluator()

        for name, algorithm, hyper_param, iterations in zip(parameters['names'], parameters['algorithms'],
                                                            parameters['hyperparameters'], parameters['iterations']):

            print("Starting model {}".format(name))
            ml_pipe = Pipeline([('scaler', StandardScaler()),
                                ('pca', PCA()),
                                ('algo', algorithm)])

            hyper_param = {**pca_params, **hyper_param}

            random_search = RandomizedSearchCV(ml_pipe, param_distributions=hyper_param, n_iter=iterations,
                                               cv=5, scoring=evaluator, n_jobs=n_jobs, verbose=1)
            random_search.fit(self._x_train, self._y_train)

            models[name] = (round(random_search.best_score_, 3), random_search)

        models = self._models_sorter(models)
        self._models = models

        print('The machine learning battle has finished')

    def predict(self) -> np.ndarray:
        """
        Predicts the results with the best model issue form the optimization process
        :return: predictions
        """
        y_pred = []
        under_value = 0
        upper_values = list()
        upper_values.extend(range(10000, self._x_test.shape[0], 10000))
        upper_values.append(self._x_test.shape[0] + 1)

        for upper_value in upper_values:
            y_pred_temp = self._models[0][1][1].best_estimator_.predict(self._x_test[under_value: upper_value])
            y_pred = np.append(y_pred, y_pred_temp)
            under_value = upper_value

        return y_pred

    # Internal methods
    def _load_parameters(self) -> dict:
        """
        Loads the parameters to be used in the machine learning battle
        :return: dictionary with all the parameter for the required model
        """

        if self._category is 'supervised' and self._goal is 'classification':
            from sklearn.neural_network import MLPClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

            algorithm_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest",
                               "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]

            algorithms = [KNeighborsClassifier(), SVC(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(),
                          MLPClassifier(), AdaBoostClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis()]

            algorithm_params = [
                {'algo__n_neighbors': randint(2, 7), 'algo__weights': ['uniform', 'distance'],
                 'algo__leaf_size': randint(20, 40)},  # KNN
                {'algo__C': expon(scale=1.0), 'algo__kernel': ['linear']},  # SCV - Linear
                {'algo__C': expon(scale=1.0), 'algo__gamma': expon(scale=1.0)},  # SVC - RBF
                {'algo__max_depth': randint(2, 7), 'algo__min_samples_split': randint(2, 7),
                 'algo__max_features': ['auto', 'sqrt', 'log2']},  # Decision Tree
                {'algo__max_depth': randint(2, 7), 'algo__min_samples_split': randint(2, 7),
                 'algo__max_features': ['auto', 'sqrt', 'log2']},  # Random Forest
                {'algo__activation': ['tanh', 'logistic', 'relu'], 'algo__alpha': expon(scale=0.0001),
                 'algo__learning_rate': ['constant', 'invscaling', 'adaptive']},  # Multilayer Perceptron
                {'algo__learning_rate': expon(scale=1.0)},  # Adaboost Classifier
                {},  # Naive Bayes
                {'algo__tol': expon(scale=0.0001)}]  # QDA

            algorithm_iterations = [60, 60, 60, 60, 60, 60, 60, 1, 60]

        if self._category is 'supervised' and self._goal is 'regression':
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, BayesianRidge, ElasticNetCV,\
                PassiveAggressiveRegressor, TheilSenRegressor, SGDRegressor
            from sklearn.neural_network import MLPRegressor
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.svm import SVR
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

            algorithm_names = ["Linear Regression",  "Linear Lars", "Linear Ridge", "Linear Lasso", "BayesianRidge",
                               "ElasticNetCV", "PassiveAggressive Regressor", "TheilSen Regressor", "SGD Regressor",
                               "K-Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest",
                               "Gradient Boosting", "Neural Net", "AdaBoost"]

            algorithms = [LinearRegression(), Lars(), Ridge(), Lasso(), BayesianRidge(), ElasticNetCV(),
                          PassiveAggressiveRegressor(), TheilSenRegressor(), SGDRegressor(), KNeighborsRegressor(),
                          SVR(), SVR(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(),
                          MLPRegressor(), AdaBoostRegressor()]

            algorithm_params = [
                {},  # Linear Regression
                {},  # Linear Lars Regression
                {'rgs__alpha': uniform(1e-9, 1)},  # Linear Ridge Regression
                {'rgs__alpha': uniform(1e-9, 1)},  # Linear Lasso Regression
                {'rgs__n_iter': randint(300, 600), 'rgs__tol': uniform(1e-5, 1e-5),
                 'rgs__alpha_1': uniform(1e-9, 1e-3), 'rgs__alpha_2':  uniform(1e-9, 1e-3),
                 'rgs__lambda_1':  uniform(1e-9, 1e-3), 'rgs__lambda_2':  uniform(1e-9, 1e-3)},
                # Bayesian Rigde Regression
                {'rgs__l1_ratio': uniform(0, 1.0), 'rgs__n_alphas': randint(2, 6 + 1), 'rgs__tol': uniform(1e-3, 0.21)},
                # ElasticNetCV
                {'rgs__tol': uniform(1e-3, 0.21), 'rgs__epsilon': uniform(0.1, 2.5)},  # PassiveAggressive Regression
                {'rgs__max_subpopulation': randint(5e3, 5e4), 'rgs__tol': uniform(5.e-4, 5.e-3)},  # TheilSen Regression
                {'rgs__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber', 'squared_loss'],
                 'rgs__alpha': uniform(5e-5, 5e-4), 'rgs__penalty': ['none', 'l2', 'l1', 'elasticnet'],
                 'rgs__max_iter': [1000],
                 'rgs__tol': uniform(5e-4, 5e-3)},  # SGD Regression
                {'rgs__n_neighbors': randint(2, 15), 'rgs__weights': ['uniform', 'distance'],
                 'rgs__leaf_size': randint(20, 40)},  # KNN
                {'rgs__C': uniform(0.1, 2.0), 'rgs__kernel': ['linear']},  # SVR - Linear
                {'rgs__C': uniform(0.1, 2.0), 'rgs__gamma': uniform(0.1, 2.5), 'rgs__kernel': ['rbf', 'sigmoid']},
                # SVR - RBF
                {'rgs__max_depth': randint(2, 8), 'rgs__min_samples_split': randint(2, 8),
                 'rgs__max_features': ['auto', 'sqrt', 'log2'], 'rgs__criterion': ['mse', 'mae']},  # Decision Tree
                {'rgs__max_depth': randint(2, 8), 'rgs__min_samples_split': randint(2, 8),
                 'rgs__max_features': ['auto', 'sqrt', 'log2'], 'rgs__criterion': ['mse', 'mae']},  # Random Forest
                {'rgs__max_depth': randint(2, 8), 'rgs__min_samples_split': randint(2, 8),
                 'rgs__max_features': ['auto', 'sqrt', 'log2'], 'rgs__criterion': ['mae', 'friedman_mse'],
                 'rgs__loss': ['ls', 'lad', 'huber']},  # Gradient Boosting
                {'rgs__activation': ['tanh', 'logistic', 'relu'], 'rgs__alpha': uniform(5e-5, 5e-4),
                 'rgs__learning_rate': ['constant', 'invscaling', 'adaptive'], 'rgs__early_stopping': [True],
                 "rgs__max_iter": [1000], "rgs__early_stopping ": [True]},  # MLP Regression
                {'rgs__learning_rate': uniform(1e-3, 5e-2),
                 'rgs__loss': ['linear', 'square', 'exponential']}]  # Adaboost Regression

            algorithm_iterations = [1] * 2 + [60] * (len(algorithm_params) - 2)

        return {'names': algorithm_names, 'algorithms': algorithms, 'hyperparameters': algorithm_params,
                'iterations': algorithm_iterations}

    def _evaluator(self) -> str:
        """
        Selects the evaluator taking in consideration the problem we are solving
        :return:
        """
        if self._category is 'supervised' and self._goal is 'classification':
            return 'accuracy'

        if self._category is 'supervised' and self._goal is 'regression':
            return 'neg_mean_squared_error'

    def _models_sorter(self, models: dict) -> list:
        """
        Selects the evaluator taking in consideration the problem we are solving
        :return:
        """
        if self._category is 'supervised' and self._goal is 'classification':
            return sorted(models.items(), key=lambda kv: kv[1], reverse=True)  # Higher is better

        if self._category is 'supervised' and self._goal is 'regression':
            return sorted(models.items(), key=lambda kv: kv[1], reverse=False)  # Lower is better
