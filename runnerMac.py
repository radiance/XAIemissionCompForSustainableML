import sys
import pathlib
import os
from codecarbon import EmissionsTracker
import timeit

## GLIOMA Dataset

# # CLASSIFICATION
classification_path = str(pathlib.Path(__file__).parent.resolve()) + "/glioma/scripts/Classification"
sys.path.append(classification_path)
os.chdir(classification_path)

with EmissionsTracker() as tracker:
    start = timeit.default_timer()
    import main
    main.main()
    stop = timeit.default_timer()
    print('Runtime: ', stop - start)

print('for vizualization, call: python "./codecarbon-viz/carbonboard.py" --filepath="' + classification_path + '/emissions.csv"')

sys.path.remove(classification_path)
sys.modules.pop('main')

## EnergyEfficiencyDataset

path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve().parent.resolve()) + "/testEnergyEfficiencyDataset"
sys.path.append(path)
os.chdir(path)

with EmissionsTracker() as tracker:
    start = timeit.default_timer()
    import main
    main.main()
    stop = timeit.default_timer()
    print('Runtime: ', stop - start)  

print("for vizualization, call: python './codecarbon-viz/carbonboard.py' --filepath='" + path + "/emissions.csv'")

sys.path.remove(path)
sys.modules.pop('main')

## YOLO

yolo_path = str(pathlib.Path(__file__).parent.resolve().parent.resolve()) + "/testYOLO/yolov5"
print(yolo_path)
sys.path.append(yolo_path)
os.chdir(yolo_path)

with EmissionsTracker() as tracker:
    start = timeit.default_timer()
    import main
    stop = timeit.default_timer()
    print('Runtime: ', stop - start)  

print("for vizualization, call: python './codecarbon-viz/carbonboard.py' --filepath='" + yolo_path + "/emissions.csv'")

sys.path.remove(yolo_path)
sys.modules.pop('main')
