import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
pd.options.mode.chained_assignment = None
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

def clean(data):
    for col_name in data.columns:
        data_col = data[col_name]
        
        q1 = data_col.quantile(0.25)
        q3 = data_col.quantile(0.75)
        iqr = q3 - q1

        outside_iqr = (data_col < (q1 - 3 * iqr)) | (data_col > (q3 + 3 * iqr))
        new_col_mean = data_col[~outside_iqr].mean()

        data[col_name][outside_iqr] = new_col_mean
        
    return data

def test(X, y, classifier_cls):
    clf = classifier_cls(random_state=0)
    scores = cross_val_score(clf, X, y)
    return scores.mean()

def load_train(section):
    train = pd.read_csv('1.{0}/TrainData{0}.txt'.format(section), delim_whitespace=True, header=None)
    labels = pd.read_csv('1.{0}/TrainLabel{0}.txt'.format(section), header=None)
    labels = labels.rename(columns={0: "label"})
    
    return train, labels

def load_test(section):
    return pd.read_csv('1.{0}/TestData{0}.txt'.format(section), delim_whitespace=True, header=None)

def find_best_classifier(section, classifiers):
    train, labels = load_train(section)
    cleaned_train = clean(train)
    X, y = cleaned_train, labels.iloc[:,0]

    best_classifier = classifiers[0]
    best_score = test(X, y, best_classifier)
    
    for curr_classifier in classifiers:
        curr_score = test(X, y, curr_classifier)
        
        print('Section 1.{} with classifier {} got score: {}'.format(section, curr_classifier.__name__, curr_score))
        
        if curr_score > best_score:
            best_score = curr_score
            best_classifier = curr_classifier
    
    print('***\nBest classifier found for section 1.{} is {} with score: {}\n***'.format(section, best_classifier.__name__, best_score))
    clf = best_classifier(random_state=0)
    return clf.fit(X, y)

def predict(section, model):
    test = load_test(section)
    cleaned_test = clean(test)
    predictions = model.predict(cleaned_test)
    pd.Series(predictions).to_csv('scikit_predictions{}.txt'.format(section), sep='\t', index=False, header=False)
    
    return predictions

sections = list(range(1, 6))
classifiers = [RandomForestClassifier, SVC, AdaBoostClassifier]

for section in sections:
    model = find_best_classifier(section, classifiers)
    predict(section, model)