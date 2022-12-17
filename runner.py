import sys
import pathlib
import os
from codecarbon import OfflineEmissionsTracker
import timeit
from sys import platform
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns


## GLIOMA Dataset

# # CLUSTERING - Not needed!
# clustering_path = str(pathlib.Path(__file__).parent.resolve()) + "\\glioma\\scripts\\Clustering"
# sys.path.append(clustering_path)
# os.chdir(clustering_path)
#
# with EmissionsTracker() as tracker:
#     start = timeit.default_timer()
#     import main
#     stop = timeit.default_timer()
#     print('Runtime ' + project_name + ': ', stop - start)  
#
# sys.path.remove(clustering_path)
# sys.modules.pop('main')


# # GLIOMA CLASSIFICATION
def gliomaClassification(do_shap, run_number=0, do_feature_reduction=False):
    origin_path = path = str(pathlib.Path(__file__).parent.resolve())
    path = str(pathlib.Path(__file__).parent.resolve()) + "\\glioma\\scripts\\Classification"
    if platform == "darwin" or platform == "linux":
        path = path.replace("\\", "/")
    sys.path.append(path)
    os.chdir(path)

    project_name = "glioma"
    if do_shap:
        project_name += "_shap"
    if do_feature_reduction:
        project_name += "_reduction"
    project_name += "_" + str(run_number)
    tracker = OfflineEmissionsTracker(country_iso_code="AUT", project_name=project_name, output_dir=root_dir)
    tracker.start()
    start = timeit.default_timer()
    import main
    main.main(do_shap, do_feature_reduction)
    stop = timeit.default_timer()
    print('Runtime ' + project_name + ': ', stop - start)
    emissions_glioma = tracker.stop()
    os.chdir(origin_path)
    sys.path.remove(path)
    sys.modules.pop('main')


# ## EnergyEfficiencyDataset
def energyEfficiencyDataset(do_shap, run_number=0, do_feature_reduction=False):
    origin_path = path = str(pathlib.Path(__file__).parent.resolve())
    path = str(pathlib.Path(__file__).parent.resolve()) + "\\testEnergyEfficiencyDataset"
    path = path.replace("\\", "/")
    sys.path.append(path)
    os.chdir(path)

    project_name = "EnergyEfficiencyDataset"
    if do_shap:
        project_name += "_shap"
    if do_feature_reduction:
        project_name += "_reduction"
    project_name += "_" + str(run_number)
    tracker = OfflineEmissionsTracker(country_iso_code="AUT", project_name=project_name, output_dir=root_dir)
    tracker.start()
    start = timeit.default_timer()
    import main
    main.main(do_shap, do_feature_reduction)
    stop = timeit.default_timer()
    print('Runtime ' + project_name + ': ', stop - start)
    emissions_energy_energy_efficiency_dataset = tracker.stop()
    os.chdir(origin_path)
    sys.path.remove(path)
    sys.modules.pop('main')


# Yolov5
def yolov5(do_shap, run_number=0, do_feature_reduction=False):
    origin_path = path = str(pathlib.Path(__file__).parent.resolve())
    path = str(pathlib.Path(__file__).parent.resolve()) + "\\testYOLO\\yolov5"
    if platform == "darwin" or platform == "linux":
        # path = str(pathlib.Path(__file__).parent.resolve().parent.resolve()) + "\\testYOLO\\yolov5"
        path = path.replace("\\", "/")
    sys.path.append(path)
    os.chdir(path)

    project_name = "yolov5"
    if do_shap:
        project_name += "_shap"
    if do_feature_reduction:
        project_name += "_reduction"
    project_name += "_" + str(run_number)
    tracker = OfflineEmissionsTracker(country_iso_code="AUT", project_name=project_name, output_dir=root_dir)
    tracker.start()
    start = timeit.default_timer()
    import main
    main.main(do_shap, do_feature_reduction)
    stop = timeit.default_timer()
    print('Runtime ' + project_name + ': ', stop - start)
    emissions_yolo_v5 = tracker.stop()
    os.chdir(origin_path)
    sys.path.remove(path)
    sys.modules.pop('main')


def create_box_plots(df, column, title="", y_axis_label="", filter=""):
    if title == "":
        title = column
    df = df[[column]]

    if filter != "":
        df = df.filter(like=filter, axis=0)
        df.index = df.index.str.replace(re.escape(filter) + r"_", "", regex=True)
        df.index = df.index.str.replace(filter, "regular")

    # TODO: convert energy consumption to W/h instead of kW/h?
    # emissions_df["emissions"] = emissions_df["emissions"] * 1000

    sns.set_style("whitegrid")
    plot = sns.boxplot(x=df.index, y=column, data=df)

    plot.set_title(title)
    plot.set_xlabel("")
    plt.suptitle(filter)
    plot.set_ylabel(y_axis_label)

    # rotate labels for overall plots
    label_rotation = 0 if filter != "" else 90
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=label_rotation)

    if filter != "":
        plt.savefig(column + '_' + filter + '.png', bbox_inches="tight")
    else:
        plt.savefig(column + '.png', bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    file = "emissions.csv"
    root_dir = os.getcwd()
    if os.path.exists(file):
        os.remove(file)

    # configure number of runs here
    number_of_runs = 3

    for run_number in range(1, number_of_runs + 1):
        print("Starting Run #" + str(run_number))
        for do_shap in [True, False]:
            gliomaClassification(do_shap, run_number)
            energyEfficiencyDataset(do_shap, run_number)
            yolov5(do_shap, run_number)

        # Feature Reduction
        gliomaClassification(do_shap=False, run_number=run_number, do_feature_reduction=True)
        energyEfficiencyDataset(do_shap=False, run_number=run_number, do_feature_reduction=True)
        yolov5(do_shap=False, run_number=run_number, do_feature_reduction=True)

    print("Execution Finished.")

    print("Preparing Emission Data")
    emissions_df = pd.read_csv(file)
    emissions_df.set_index("project_name", inplace=True)

    # drop indices from runs names
    emissions_df.index = emissions_df.index.str.replace(r"_\d+", "", regex=True)

    print("Creating Plots")
    create_box_plots(emissions_df, "energy_consumed", title="Consumed Energy", y_axis_label="kW/h")
    create_box_plots(emissions_df, "duration", title="Duration", y_axis_label="seconds")
    create_box_plots(emissions_df, "emissions", title="Carbon Emissions", y_axis_label="kg")
    create_box_plots(emissions_df, "energy_consumed", title="Consumed Energy", y_axis_label="kW/h", filter="glioma")
    create_box_plots(emissions_df, "duration", title="Duration", y_axis_label="seconds", filter="glioma")
    create_box_plots(emissions_df, "emissions", title="Carbon Emissions", y_axis_label="kg", filter="glioma")
    create_box_plots(emissions_df, "energy_consumed", title="Consumed Energy", y_axis_label="kW/h",
                     filter="EnergyEfficiencyDataset")
    create_box_plots(emissions_df, "duration", title="Duration", y_axis_label="seconds",
                     filter="EnergyEfficiencyDataset")
    create_box_plots(emissions_df, "emissions", title="Carbon Emissions", y_axis_label="kg",
                     filter="EnergyEfficiencyDataset")
    create_box_plots(emissions_df, "energy_consumed", title="Consumed Energy", y_axis_label="kW/h", filter="yolov5")
    create_box_plots(emissions_df, "duration", title="Duration", y_axis_label="seconds", filter="yolov5")
    create_box_plots(emissions_df, "emissions", title="Carbon Emissions", y_axis_label="kg", filter="yolov5")

    # TODO:
    # Perform any necessary statistical calculations and print some fancy results
    # Use "emissions_df" as data. Depending on the goal, it might be necassary to filter for specific items.
    # This code could become handy again. Feel free to reuse:
    #   filter = "glioma/yolov5/EnergyEfficiencyDataset"
    #   emissions_df = emissions_df.filter(like=filter, axis=0)
    #   emissions_df.index = emissions_df.index.str.replace(re.escape(filter) + r"_", "", regex=True)
    #   emissions_df.index = emissions_df.index.str.replace(filter, "regular")

    print("Performing statistical calculations")

    print('for further vizualization, call: python "./codecarbon-viz/carbonboard.py" --filepath="' + file + '"')
