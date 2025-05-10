import os
import pickle

import numpy as np
import pandas as pd
from natsort import natsorted

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
    'addressbook': "addressbook-svm-rbf-doc2vec-distance-content-tags.sav",
    'claroline': "claroline-svm-rbf-doc2vec-distance-content-tags.sav",
    'dimeshift': "dimeshift-svm-rbf-doc2vec-distance-content-tags.sav",
    'mantisbt': "mantisbt-svm-rbf-doc2vec-distance-content-tags.sav",
    'mrbs': "mrbs-svm-rbf-doc2vec-distance-content-tags.sav",
    'pagekit': "pagekit-svm-rbf-doc2vec-distance-content-tags.sav",
    'petclinic': "petclinic-svm-rbf-doc2vec-distance-content-tags.sav",
    'phoenix': "phoenix-svm-rbf-doc2vec-distance-content-tags.sav",
    'ppma': "ppma-svm-rbf-doc2vec-distance-content-tags.sav",
}

if __name__ == '__main__':
    for setting in ["within-apps", "across-apps"]:
        print("Setting :  " + setting)
        for app in APPS:
            print("App :  " + setting)
            comparison_df = None
            classifier_path = CLASSIFIERS[app]
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
            elif 'doc2vec-distance-content-tags' in classifier:
                column = 'doc2vec-distance-content-tags'
            elif 'doc2vec-distance-tags' in classifier:
                column = 'doc2vec-distance-tags'
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

            items = natsorted(set.union(set([item[0] for item in lis]), set([item[1] for item in lis])))

            value = dict(zip(items, range(len(items))))
            dist_matrix = np.zeros((len(items), len(items)))

            for i in range(len(lis)):
                # upper triangle
                dist_matrix[value[lis[i][0]], value[lis[i][1]]] = lis[i][2]
                # lower triangle
                dist_matrix[value[lis[i][1]], value[lis[i][0]]] = lis[i][2]

            os.makedirs(f'{base_path}/resources/csv_results_table', exist_ok=True)
            new_ss = pd.DataFrame(dist_matrix, columns=items, index=items)
            new_ss.to_csv(f'{base_path}/resources/csv_results_table/SS_as_distance_matrix_{setting}-{app}-webembed-content_tags.csv') # Distance matrix with predictions
