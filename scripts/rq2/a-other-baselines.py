import pickle
import numpy as np
import pandas as pd
from natsort import natsorted
import os


def is_clone(model, distance):
    try:
        prediction = model.predict(np.array(distance).reshape(1, -1))  # 0 = distinct, 1 = near duplicate
    except ValueError:
        prediction = [0]

    if prediction == [1]:
        return True
    else:
        return False

base_path    = os.getcwd()

APPS = ['addressbook', 'claroline', 'dimeshift', 'mantisbt', 'mrbs', 'pagekit', 'petclinic', 'phoenix', 'ppma']

CLASSIFIERS = {
    'addressbook': ["addressbook-svm-rbf-dom-rted.sav",
                    "addressbook-svm-rbf-visual-pdiff.sav"],

    'claroline': [  "claroline-svm-rbf-dom-rted.sav",
                    "claroline-svm-rbf-visual-pdiff.sav"],

    'dimeshift': [  "dimeshift-svm-rbf-dom-rted.sav",
                    "dimeshift-svm-rbf-visual-pdiff.sav"],

    'mantisbt': [   "mantisbt-svm-rbf-dom-rted.sav",
                    "mantisbt-svm-rbf-visual-pdiff.sav"],

    'mrbs':     [    "mrbs-svm-rbf-dom-rted.sav",
                    "mrbs-svm-rbf-visual-pdiff.sav"],

    'pagekit': [    "pagekit-svm-rbf-dom-rted.sav",
                    "pagekit-svm-rbf-visual-pdiff.sav"],

    'petclinic': [  "petclinic-svm-rbf-dom-rted.sav",
                    "petclinic-svm-rbf-visual-pdiff.sav"],

    'phoenix': [    "phoenix-svm-rbf-dom-rted.sav",
                    "phoenix-svm-rbf-visual-pdiff.sav"],

    'ppma': [       "ppma-svm-rbf-dom-rted.sav",
                    "ppma-svm-rbf-visual-pdiff.sav"],
}

OUTPUT_CSV = False

if __name__ == '__main__':
    for setting in ["within-apps", "across-apps"]:
        print("Setting :  " + setting)

        for app in APPS:
            print("App :  " + setting)
            comparison_df = None
            for feature in np.arange(2):
                classifier_path = CLASSIFIERS[app][feature]
                classifier = f"{base_path}/baseline-models/{setting}-{classifier_path}"

                column = None
                if 'dom-rted' in classifier:
                    column = 'dom-rted'.upper()
                elif 'visual-hyst' in classifier:
                    column = 'VISUAL_Hyst'
                elif 'visual-pdiff' in classifier:
                    column = 'VISUAL-PDiff'
                elif 'doc2vec-distance-content' in classifier:
                    column = 'doc2vec-distance-content'
                elif 'doc2vec-distance-tags' in classifier:
                    column = 'doc2vec-distance-tags'
                elif 'doc2vec-distance-content-tags' in classifier:
                    column = 'doc2vec-distance-content-tags'
                elif 'doc2vec-distance-all' in classifier:
                    column = 'doc2vec-distance-all'

                ss = pd.read_csv(f"{base_path}/resources/baseline-dataset/SS_threshold_set.csv",
                                 usecols=['appname', 'state1', 'state2', column.replace('-', '_')])
                ss = ss.query("appname == @app")
                ss = ss.drop(['appname'], axis=1)

                model = None
                try:
                    model = pickle.load(open(classifier, 'rb'))
                except FileNotFoundError:
                    print("Cannot find classifier %s" % classifier)
                    exit()
                except pickle.UnpicklingError:
                    print(classifier)
                    exit()

                # convert distances to similarities
                ss[column.replace('-', '_')] = ss[column.replace('-', '_')].map(lambda dist: is_clone(model, dist))

                tuples = [tuple(x) for x in ss.to_numpy()]

                lis = tuples

                # get the sorted unique state list
                items = natsorted(set.union(set([item[0] for item in lis]), set([item[1] for item in lis])))
                # states with its index
                value = dict(zip(items, range(len(items))))
                dist_matrix = np.zeros((len(items), len(items)))

                for i in range(len(lis)):
                    # upper triangle
                    # lis 0 - state1, lis 1 - state2, lis 2 - is clone boolean
                    dist_matrix[value[lis[i][0]], value[lis[i][1]]] = lis[i][2]
                    # lower triangle
                    dist_matrix[value[lis[i][1]], value[lis[i][0]]] = lis[i][2]

                new_ss = pd.DataFrame(dist_matrix, columns=items, index=items)
                new_ss.to_csv(f'{base_path}/resources/csv_results_table/SS_as_distance_matrix_{setting}-{app}-{"rted" if feature==0 else "pdiff"}.csv') # Distance matrix with predictions