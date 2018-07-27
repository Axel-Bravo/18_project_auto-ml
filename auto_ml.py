import pandas as pd
import numpy as np

from scipy.stats import randint, expon

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib


class MlSupervised(object):
    """Parent class for classification and regression problems"""
    pass

class MlParameters(object):
    """
    Class used to construct the model parameters used in the AutoClf class
    """
    pass

class AutoClf (MlSupervised):
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

        pca_params = {'pca__n_components': randint(3, 10)}

        classifiers_params = [
            {'clf__n_neighbors': randint(2, 7), 'clf__weights': ['uniform', 'distance'],
             'clf__leaf_size': randint(20, 40)},  # KNN
            {'clf__C': expon(scale=1.0), 'clf__kernel': ['linear']},  # SCV - Linear
            {'clf__C': expon(scale=1.0), 'clf__gamma': expon(scale=1.0)},  # SVC - RBF
            {'clf__max_depth': randint(2, 7), 'clf__min_samples_split': randint(2, 7),
             'clf__max_features': ['auto', 'sqrt', 'log2']},  # Decision Tree
            {'clf__max_depth': randint(2, 7), 'clf__min_samples_split': randint(2, 7),
             'clf__max_features': ['auto', 'sqrt', 'log2']},  # Random Forest
            {'clf__activation': ['tanh', 'logistic', 'relu'], 'clf__alpha': expon(scale=0.0001),
             'clf__learning_rate': ['constant', 'invscaling', 'adaptive']},  # Multilayer Perceptron
            {'clf__learning_rate': expon(scale=1.0)},  # Adaboost Classifier
            {},  # Naive Bayes
            {'clf__tol': expon(scale=0.0001)}]  # QDA

        classifiers_iterations = [60, 60, 60, 60, 60, 60, 60, 1, 60]
        models = {}

        for clf_name, clf, clf_parameters, clf_iter in zip(classifiers_names, classifiers, classifiers_params,
                                                           classifiers_iterations):
            print("Starting model {}".format(clf_name))
            pipe_ML = Pipeline([('scaler', StandardScaler()),
                                ('pca', PCA()),
                                ('clf', clf)])

            clf_parameters = {**pca_params, **clf_parameters}

            random_search = RandomizedSearchCV(pipe_ML, param_distributions=clf_parameters, n_iter=clf_iter,
                                               cv=5, scoring='accuracy',
                                               n_jobs=2, verbose=1)
            random_search.fit(X_train, y_train)

            models[clf_name] = (round(random_search.best_score_, 3), random_search)

        models = sorted(models.items(), key=lambda kv: kv[1], reverse=True)
        joblib.dump(models, 'models/007 - Classifiers_Battle.pkl', compress=1)
        print('The machine learning battle has finished')
