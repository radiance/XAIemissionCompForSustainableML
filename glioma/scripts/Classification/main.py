from Classifier import Classifier
from DataManager import DataManager
import glob


def main(do_shap=True, do_feature_reduction=False):
    datamanager = DataManager()
    labels, data = datamanager.get_labels_and_data()
    classfier = Classifier(labels, data)
    classfier.predict(do_shap=do_shap, do_feature_reduction=do_feature_reduction)


