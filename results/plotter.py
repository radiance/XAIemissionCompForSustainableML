import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

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
