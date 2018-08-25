import pandas as pd
from scipy.stats import randint, expon, uniform

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
    """

    __slots__ = ['_name', '_train', '_test', '_parameters', '_category', '_goal', '_models']

    def __init__(self, name: str, category: str = 'supervised', goal: str = 'classification'):
        self._name: str = name
        self._train: pd.DataFrame = None
        self._test: pd.DataFrame = None
        self._parameters: dict = None
        self._category: str = category
        self._goal: str = goal
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

                    assert_alert = name + ' data has no "Y" column'
                    assert ("Y" in temp.columns), assert_alert

                    if name is "Train":
                        self._train = temp
                    else:
                        self._test = temp

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
    def optimize(self, n_jobs: int = 2):
        """
        Realizes the battle model optimization
        :return:
        """

        y_train = self._train['Y']
        X_train = self._train.drop(columns=['Y'])
        models = {}

        self._load_parameters()
        pca_params = {'pca__n_components': randint(2, self._train.shape[1])}
        evaluator = self._evaluator()

        for name, algo, hyper_param, iter in zip(self._parameters['names'], self._parameters['algorithms'],
                                                 self._parameters['hyperparameters'], self._parameters['iterations']):

            print("Starting model {}".format(name))
            ml_pipe = Pipeline([('scaler', StandardScaler()),
                                ('pca', PCA()),
                                ('algo', algo)])

            hyper_param = {**pca_params, **hyper_param}

            random_search = RandomizedSearchCV(ml_pipe, param_distributions=hyper_param, n_iter=iter,
                                               cv=5, scoring=evaluator, n_jobs=n_jobs, verbose=1)
            random_search.fit(X_train, y_train)

            models[name] = (round(random_search.best_score_, 3), random_search)


        models = sorted(models.items(), key=lambda kv: kv[1], reverse=True)  # Sorting models by scoring
        self._models = models
        print('The machine learning battle has finished')

    def predict(self) -> pd.DataFrame:
        """
        Predicts the results with the best model issue form the optimization process
        :return: predictions
        """
        return self._models[0][1][1].best_estimator_.predict(self._test)

    # Internal methods
    def _load_parameters(self):
        """
        Loads the parameters to be used in the machine learning battle
        :return:
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
            from sklearn.linear_model import LinearRegression, ElasticNetCV, PassiveAggressiveRegressor, \
                TheilSenRegressor, SGDRegressor
            from sklearn.neural_network import MLPRegressor
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.svm import SVR
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

            algorithm_names = ["Linear Regression", "ElasticNetCV", "PassiveAggressive Regressor", "TheilSen Regressor",
                               "SGD Regressor", "K-Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
                               "Random Forest", "Gradient Boosting", "Neural Net", "AdaBoost"]

            algorithms = [LinearRegression(), ElasticNetCV(), PassiveAggressiveRegressor(), TheilSenRegressor(),
                          SGDRegressor(), KNeighborsRegressor(), SVR(), SVR(), DecisionTreeRegressor(),
                          RandomForestRegressor(), GradientBoostingRegressor(), MLPRegressor(), AdaBoostRegressor()]

            algorithm_params = [
                {},  # Linear Regression
                {'rgs__l1_ratio': uniform(0, 1.0), 'rgs__n_alphas': randint(2, 6 + 1), 'rgs__tol': uniform(1e-3, 0.21)},
                # ElasticNetCV
                {'rgs__tol': uniform(1e-3, 0.21), 'rgs__epsilon': uniform(0.1, 2.5)},  # PassiveAggressive Regressor
                {'rgs__max_subpopulation': randint(5e3, 5e4), 'rgs__tol': uniform(5.e-4, 5.e-3)},  # TheilSen Regressor
                {'rgs__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber', 'squared_loss'],
                 'rgs__alpha': uniform(5e-5, 5e-4), 'rgs__penalty': ['none', 'l2', 'l1', 'elasticnet'],
                 'rgs__max_iter': [1000],
                 'rgs__tol': uniform(5e-4, 5e-3)},  # SGDRegressor
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
                 "rgs__max_iter": [1000], "rgs__early_stopping ": [True]},  # MLP Regressor
                {'rgs__learning_rate': uniform(1e-3, 5e-2),
                 'rgs__loss': ['linear', 'square', 'exponential']}]  # Adaboost Regressor

            algorithm_iterations = [60] * len(algorithm_params)

        self._parameters = {'names': algorithm_names, 'algorithms': algorithms,
                            'hyperparameters': algorithm_params, 'iterations': algorithm_iterations}

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
