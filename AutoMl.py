import pandas as pd
from scipy.stats import randint, expon

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib


class AutoMl (object):
    """
    Class that agglutinates all "auto" machine learning techniques for classification problems
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

    def load_data(self, train_data, test_data):
        """
        Load data from a ".csv" file or from memory
        :param train_data: file path to train data
        :type train_data: str
        :param test_data: file path to test data
        :type test_data: str
        :return: Initialize object
        """

        if self._category is 'supervised' and self._goal is 'classification':
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

    def load_parameters(self):
        """
        s
        :return:
        """
        classifiers_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                             "Naive Bayes", "QDA"]

        classifiers = [
            KNeighborsClassifier(),
            SVC(),
            SVC(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            MLPClassifier(),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        classifiers_params = [
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

        classifiers_iterations = [60, 60, 60, 60, 60, 60, 60, 1, 60]

        self._parameters = {'names': classifiers_names, 'algorithms': classifiers,
                            'hyperparameters': classifiers_params, 'iterations': classifiers_iterations}

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

    def optimize(self, n_jobs: int = 2):
        """
        Realizes a
        :return:
        """

        y_train = self._train['Y']
        X_train = self._train.drop(columns=['Y'])
        models = {}

        pca_params = {'pca__n_components': randint(2, self._train.shape[1])}

        for name, algo, hyper_param, iter in zip(self._parameters['names'], self._parameters['algorithms'],
                                                 self._parameters['hyperparameters'], self._parameters['iterations']):

            print("Starting model {}".format(name))
            ml_pipe = Pipeline([('scaler', StandardScaler()),
                                ('pca', PCA()),
                                ('algo', algo)])

            hyper_param = {**pca_params, **hyper_param}

            random_search = RandomizedSearchCV(ml_pipe, param_distributions=hyper_param, n_iter=iter,
                                               cv=5, scoring='accuracy', n_jobs=n_jobs, verbose=1)
            random_search.fit(X_train, y_train)

            models[name] = (round(random_search.best_score_, 3), random_search)

        models = sorted(models.items(), key=lambda kv: kv[1], reverse=True)
        self._models = models
        print('The machine learning battle has finished')

    def predict(self) -> pd.DataFrame:
        """
        Predicts the results with the best model issue form the optimization process
        :return: predictions
        """
        return self._models[0][1][1].best_estimator_.predict(self._test)
