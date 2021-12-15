################################################################################
# Imports
################################################################################
import imageio
import os

import numpy as np

from sklearn                import preprocessing
from sklearn.ensemble       import RandomForestClassifier, VotingClassifier
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics        import accuracy_score
from sklearn.svm            import SVC
from sklearn.tree           import DecisionTreeClassifier

from keras.preprocessing    import image

#from deep_feature_extractor.extract_pretrained_features import get_model, next_patch

################################################################################
# Constants, definitions and global variables
################################################################################
_DATA_FOLDER_FEATURE = "./data"
_DATA_FOLDER_IMAGES = "."

_INPUT_SHAPE = (519, 480, 3)

_FEATURES = ("mobilenetv2", "resnet50v2", "vgg16")

################################################################################
# Functions and procedures
################################################################################

################################################################################
# Classes
################################################################################

################################################################################
# Main
################################################################################
if __name__ == "__main__":
    feature = None    # Auxiliar, feature loop
    entry = None      # Auxiliar, loop
    folder = None     # Auxiliar, folder loop
    f_data = None     # Feature data and labels
    buf = None        # Buffer for data
    arr = None        # Auxiliar, array
    acc = None        # Accuracy

    # Classifiers
    knn = None
    rf = None
    mlp = None
    svc = None
    dt = None
    voting = None
    
    # Reading data
    f_data = {}

    for feature in _FEATURES:
        f_data[feature] = {"x": [],
                           "y": []}
        
        for folder in [ entry.path for entry in
                        os.scandir(os.path.join(_DATA_FOLDER_FEATURE, feature)) if entry.is_dir() ]:
            filepath = "{}/1.npy".format(folder)

            if ( os.path.isfile(filepath) ):
                buf = np.load(filepath)

                for arr in buf:
                    f_data[feature]["x"].append(arr)
                    f_data[feature]["y"].append(os.path.basename(folder))

    for feature in _FEATURES:
        # model, preprocess_input = get_model(feature,
        #                                     weights='imagenet',
        #                                     include_top=False,
        #                                     input_shape=_INPUT_SHAPE,
        #                                     pooling='avg')

        f_data[feature]["x"] = preprocessing.normalize(f_data[feature]["x"], norm="l2")
        
        # Random Forest
        rf = RandomForestClassifier()
        rf.fit(f_data[feature]["x"], f_data[feature]["y"])

        print("[{}] RF Accuracy: {}".format(feature,
                                            rf.score(f_data[feature]["x"], f_data[feature]["y"])))

        # KNN
        knn = KNeighborsClassifier(n_neighbors = 5)
        knn.fit(f_data[feature]["x"], f_data[feature]["y"])
        
        print("[{}] KNN Accuracy: {}".format(feature,
                                             knn.score(f_data[feature]["x"], f_data[feature]["y"])))
        
        # MLP
        mlp = MLPClassifier()
        mlp.fit(f_data[feature]["x"], f_data[feature]["y"])

        print("[{}] MLP Accuracy: {}".format(feature,
                                             mlp.score(f_data[feature]["x"], f_data[feature]["y"])))

        # SVC
        svc = SVC()
        svc.fit(f_data[feature]["x"], f_data[feature]["y"])

        print("[{}] SVC Accuracy: {}".format(feature,
                                             svc.score(f_data[feature]["x"], f_data[feature]["y"])))
        
        # DT
        dt = DecisionTreeClassifier()
        dt.fit(f_data[feature]["x"], f_data[feature]["y"])

        print("[{}] DT Accuracy: {}".format(feature,
                                             dt.score(f_data[feature]["x"], f_data[feature]["y"])))

        # Hard Voting
        voting = VotingClassifier((("RF", rf),
                                   ("KNN", knn),
                                   ("MLP", mlp),
                                   ("SVC", svc),
                                   ("DT", dt)),
                                  voting="hard")

        voting.fit(f_data[feature]["x"], f_data[feature]["y"])
        print("[{}] H Voting Accuracy: {}".format(feature,
                                                  voting.score(f_data[feature]["x"], f_data[feature]["y"])))
        
        # Soft Voting
        voting = VotingClassifier((("RF", rf),
                                   ("KNN", knn),
                                   ("MLP", mlp),
                                   ("SVC", svc),
                                   ("DT", dt)),
                                  voting="hard")

        voting.fit(f_data[feature]["x"], f_data[feature]["y"])
        print("[{}] S Voting Accuracy: {}".format(feature,
                                                  voting.score(f_data[feature]["x"], f_data[feature]["y"])))
