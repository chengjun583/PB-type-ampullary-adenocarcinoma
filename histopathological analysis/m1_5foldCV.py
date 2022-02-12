
from pandas import read_csv
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



def load_data(filename):
    data = read_csv(filename, sep='\t')
    print(data.head(10))
    print(data.groupby('label').size())
    print(data.groupby('labelname').size())
    array = data.values
    #ID
    X = array[:, 0]
    #Feature
    y = array[:, 1:-2]
    #Label
    label = array[:, -2]
    label_name = array[:, -1]
    return X,y,label,label_name


def select_feature(feature, label, feature_number):
    selector = SelectKBest(score_func=f_classif, k=feature_number)
    feature_new = selector.fit_transform(feature, label)
    selected_feature_index = selector.get_support(True)
    return feature_new,selected_feature_index


def cross_validation(model, X_train, y_train, seed):
    kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    #print('%s: \n mean:%f (std:%f)' % (model, cv_results.mean(), cv_results.std()))
    return cv_results.mean(),cv_results.std()


def according_cv_select_model(X_train, y_train, para):
    Model = SVC(C=para, kernel='linear', probability=True, verbose=False, class_weight='balanced',random_state=7)
    # 5-fold cross validation
    mean,std = cross_validation(Model, X_train, y_train, seed=5)
    return mean,std


if __name__ == "__main__":
    filename = 'data/TCGA_CHOL_PAAD_imData.txt'
    ID, feature, label, label_name = load_data(filename)
    label = label.astype('int')

    Feature_Numbers = [15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110]
    best_cv_result = 0
    best_C = 0
    best_feature_number = 0
    f1 = open('CV_results_SVC_linear.txt', 'a')
    # dataset split
    X_train, X_test, y_train, y_test = train_test_split(feature, label_name, test_size=0.2, random_state=42)
    # grid search
    for feature_number in Feature_Numbers:
        f1.write(str(feature_number))
        f1.write('\t')
        # feature selection
        X_train_new, selected_feature_index = select_feature(X_train, y_train, feature_number)
        C = [0.1,0.3,0.5,0.7,0.9,1,3,5,7,9,10]
        for c in C:
            print("feature name:%d,C:%d"%(feature_number,c))
            mean,std = according_cv_select_model(X_train_new, y_train, para=c)
            if mean >= best_cv_result:
                best_cv_result = mean
                best_C = c
                best_feature_number = feature_number
            result = str(mean) + '(' + str(std) + ')' + '\t'
            f1.write(result)
            f1.write('\t')
        f1.write('\n')
    f1.close()

    f2 = open('best_para_SVC_linear.txt', 'a')
    f2.write('best 5-fold cv result:' + str(best_cv_result))
    f2.write('\n')
    f2.write('best C:' + str(best_C))
    f2.write('\n')
    f2.write('best feature number:' + str(best_feature_number))
    f2.write('\n')
    f2.close()

