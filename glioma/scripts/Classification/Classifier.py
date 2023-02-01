import pandas as pd
import Plotter as Plotter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

class Classifier:
    data = pd.DataFrame()
    labels = pd.DataFrame()

    def __init__(self, labels, data):
        self.data = data
        self.labels = labels

    def predict(self, do_shap=True, do_feature_reduction=False):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.20, random_state=1)
        if do_feature_reduction:
            feature_reduction(X_train, X_test, y_train, y_test)
        else:
            model = RandomForestClassifier(random_state=1)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Evaluate predictions
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))
            print("Accuracy = " + str(round(accuracy_score(y_test, predictions) * 100, 2)) + " %")
            Plotter.plot_confusion_matrix(y_test, predictions)
            if do_shap:
                Plotter.plot_prediction_desicion(model, X_test, predictions, 0)
                Plotter.plot_feature_importance_for_class(model, X_train)


def feature_reduction(X_train, X_test, y_train, y_test):
    features_ranked = feature_ranking(X_train, y_train)
    highest_score = 0
    best_features = []
    for index, row in features_ranked.iterrows():
        X_train.drop(labels=index, axis=1, inplace=True)
        X_test.drop(labels=index, axis=1, inplace=True)

        model = RandomForestClassifier(random_state=1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        model = RandomForestClassifier(random_state=1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)

        # Evaluate predictions
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        print("Accuracy = " + str(round(score * 100, 2)) + " % for " + str(len(X_train.columns)) + " features.")
        Plotter.plot_confusion_matrix(y_test, predictions)
        if score > highest_score:
            highest_score = score
            best_features = list(X_train.columns)

        if score < 0.7 or len(X_train.columns) <= 1:
            print("Highest score was: " + str(round(highest_score * 100, 2)) + " % for " + str(len(best_features)) + " features." )
            print(best_features)
            break

def feature_ranking(X, y):
    model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=5)
    rfe.fit(X, y)

    ranking = pd.DataFrame(rfe.ranking_, index=X.columns, columns=["Rank"])
    ranking.sort_values(by="Rank", inplace=True, ascending=False)
    return ranking
