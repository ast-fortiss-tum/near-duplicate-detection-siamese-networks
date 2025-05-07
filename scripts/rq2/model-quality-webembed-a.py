import csv
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

SETTINGS = ["across-apps"] # 'within-apps' or 'across-apps'
print(f'====== Setting: {SETTINGS[0]} ======')

APPS = ['addressbook', 'claroline', 'dimeshift', 'mantisbt', 'mrbs', 'pagekit', 'petclinic', 'phoenix', 'ppma']

CLASSIFIERS = {
    'addressbook': [f"trained_classifiers/{SETTINGS[0]}-addressbook-svm-rbf-doc2vec-distance-content-tags.sav",
                    f"trained_classifiers/{SETTINGS[0]}-addressbook-svm-rbf-doc2vec-distance-content.sav",
                    f"trained_classifiers/{SETTINGS[0]}-addressbook-svm-rbf-doc2vec-distance-tags.sav"],
    'claroline': [f"trained_classifiers/{SETTINGS[0]}-claroline-svm-rbf-doc2vec-distance-content-tags.sav",
                  f"trained_classifiers/{SETTINGS[0]}-claroline-svm-rbf-doc2vec-distance-content.sav",
                  f"trained_classifiers/{SETTINGS[0]}-claroline-svm-rbf-doc2vec-distance-tags.sav"],
    'dimeshift': [f"trained_classifiers/{SETTINGS[0]}-dimeshift-svm-rbf-doc2vec-distance-content-tags.sav",
                  f"trained_classifiers/{SETTINGS[0]}-dimeshift-svm-rbf-doc2vec-distance-content.sav",
                  f"trained_classifiers/{SETTINGS[0]}-dimeshift-svm-rbf-doc2vec-distance-tags.sav"],
    'mantisbt': [f"trained_classifiers/{SETTINGS[0]}-mantisbt-svm-rbf-doc2vec-distance-content-tags.sav",
                 f"trained_classifiers/{SETTINGS[0]}-mantisbt-svm-rbf-doc2vec-distance-content.sav",
                 f"trained_classifiers/{SETTINGS[0]}-mantisbt-svm-rbf-doc2vec-distance-tags.sav"],
    'mrbs': [f"trained_classifiers/{SETTINGS[0]}-mrbs-svm-rbf-doc2vec-distance-content-tags.sav",
             f"trained_classifiers/{SETTINGS[0]}-mrbs-svm-rbf-doc2vec-distance-content.sav",
             f"trained_classifiers/{SETTINGS[0]}-mrbs-svm-rbf-doc2vec-distance-tags.sav"],
    'pagekit': [f"trained_classifiers/{SETTINGS[0]}-pagekit-svm-rbf-doc2vec-distance-content-tags.sav",
                f"trained_classifiers/{SETTINGS[0]}-pagekit-svm-rbf-doc2vec-distance-content.sav",
                f"trained_classifiers/{SETTINGS[0]}-pagekit-svm-rbf-doc2vec-distance-tags.sav"],
    'petclinic': [f"trained_classifiers/{SETTINGS[0]}-petclinic-svm-rbf-doc2vec-distance-content-tags.sav",
                  f"trained_classifiers/{SETTINGS[0]}-petclinic-svm-rbf-doc2vec-distance-content.sav",
                  f"trained_classifiers/{SETTINGS[0]}-petclinic-svm-rbf-doc2vec-distance-tags.sav"],
    'phoenix': [f"trained_classifiers/{SETTINGS[0]}-phoenix-svm-rbf-doc2vec-distance-content-tags.sav",
                f"trained_classifiers/{SETTINGS[0]}-phoenix-svm-rbf-doc2vec-distance-content.sav",
                f"trained_classifiers/{SETTINGS[0]}-phoenix-svm-rbf-doc2vec-distance-tags.sav"],
    'ppma': [f"trained_classifiers/{SETTINGS[0]}-ppma-svm-rbf-doc2vec-distance-content-tags.sav",
             f"trained_classifiers/{SETTINGS[0]}-ppma-svm-rbf-doc2vec-distance-content.sav",
             f"trained_classifiers/{SETTINGS[0]}-ppma-svm-rbf-doc2vec-distance-tags.sav"],
}

OUTPUT_CSV = True
OUTPUT_DIR = 'csv_results_table'
filename = f'{OUTPUT_DIR}/rq2-original-{SETTINGS[0]}.csv'

if __name__ == '__main__':
    os.chdir("../..")

    if OUTPUT_CSV:
        # Create the directory if it does not exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # create csv file to store the results
        if not os.path.exists(filename):
            header = ['Setting', 'App', 'Feature', 'Classifier', 'Precision', 'Recall', 'F1']
            with open(filename, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)

    for app in APPS:
        print(app)
        comparison_df = None
        for feature in np.arange(3): # 0 = content-tags, 1 = content, 2 = tags
            classifier = CLASSIFIERS[app][feature]
            print(classifier)

            if OUTPUT_CSV:
                comparison_df = pd.read_csv(filename)

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

            ss = pd.read_csv('resources/baseline-dataset/SS_threshold_set.csv',
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

            new_ss = pd.DataFrame(dist_matrix, columns=items, index=items)
            new_ss.to_csv(f'prediction_matrix/SS_as_distance_matrix_{SETTINGS[0]}-{app}-webembed-{"content_tags" if feature==0 else "content" if feature==1 else "tags"}.csv') # Distance matrix with predictions