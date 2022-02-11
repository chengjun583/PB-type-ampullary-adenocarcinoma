
###This code is used for Machine learning :
    # SVC
    # LogisticRegression
    # KNeighborsClassifier
    # RandomForestClassifier

import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, f_regression, chi2, f_classif
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve,confusion_matrix
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import  classification_report
from sklearn import metrics
import pandas as pd
import os

print("Sklearn verion is {}".format(sklearn.__version__))

def load_data(filename):
    data = pd.read_csv(filename, sep='\t')
    print(data.head(10))
    print(data.groupby('label').size())
    print(data.groupby('labelname').size())
    columns_name = data.columns
    fea_name = [ ]
    for i in range(1,151):
        fea_name.append(columns_name[i])

    array = data.values
    #ID
    X = array[:, 0]
    #Feature
    y = array[:, 1:-2]
    #Label
    label = array[:, -2]
    label_name = array[:, -1]
    return X,y,label,label_name,fea_name


def load_some_Column(feature, feature_index):
    feature_new = feature[:,feature_index]
    print('feature new shape:', feature_new.shape)
    return feature_new


def select_feature(feature, label, feature_number):
    selector = SelectKBest(score_func=f_classif, k=feature_number)
    feature_new = selector.fit_transform(feature, label)
    selected_feature_index = selector.get_support(True)
    return feature_new,selected_feature_index


def train_model(X_train, y_train,C_para):
    Model = SVC(C=C_para, kernel='linear', gamma='auto', probability=True,
                verbose=False, class_weight='balanced',random_state=7).fit(X_train, y_train,sample_weight=None)
    return Model


def get_model_selected_fea_index(coefs):
    fea_cols = []
    for i in range(len(coefs[0])):
        if coefs[0][i] != 0:
            fea_cols.append(i)
    print('model selected feature index:', fea_cols)
    return fea_cols


def test_model(Model, X_test, y_test, pos_labels, savedir):
    y_pre = Model.predict(X_test)
    y_proba = Model.predict_proba(X_test)        # Probability estimates

    score = Model.score(X_test, y_test)
    Model_accuracy_score = accuracy_score(y_test, y_pre)
    Model_precision_score = precision_score(y_test, y_pre, pos_label=pos_labels)
    Model_recall_score = recall_score(y_test, y_pre, pos_label=pos_labels)
    Model_f1_score = f1_score(y_test, y_pre, pos_label=pos_labels)
    print('accuracy_score: %f,precision_score: %f,recall_score: %f,f1_score: %f'
          % (Model_accuracy_score, Model_precision_score, Model_recall_score, Model_f1_score))

    Sensitivity,Specificity = calculate_metric(y_test, y_pre)    #calculate Sensitivity and Specificity

    # Save results
    test_file = savedir + 'test_score.txt'
    f = open(test_file, 'a')
    allscore = ('modelname: SVC_linear\taccuracy_score: %f\tprecision_score: %f\trecall_score: %f\t'
                'f1_score: %f\tSensitivity: %f\tSpecificity: %f\n'
                % (Model_accuracy_score, Model_precision_score, Model_recall_score,
                   Model_f1_score,Sensitivity,Specificity))
    f.write(allscore)
    f.close()
    # Calculate confusion matrix
    C = confusion_matrix(y_test, y_pre, labels=None, sample_weight=None)
    print('confusion_matrix:', C)

    return y_proba


def AAC_test_model(Model, X_test):
    y_proba = Model.predict_proba(X_test)
    print('y_proba:',y_proba)
    return y_proba


def plot_ROC_curve(y_test, y_proba, dirname):
    fpr, tpr, threasholds = metrics.roc_curve(y_test, y_proba[:, 1], pos_label='PAAD',drop_intermediate=False)  #pos_label='PAAD'
    auc = metrics.auc(fpr, tpr)
    print('AUC:',auc)

    plt.title("roc_curve of %s(AUC=%.4f)" % ('SVC-linear', auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", color="k")
    #hospital data,TCGA data
    #plt.text(0.52, 0.1, "training:TCGA data(80%)", size = 12, alpha = 1, color = "r",bbox = dict(facecolor = "r", alpha = 0.1))
    #plt.text(0.52, 0, "test:TCGA data(20%)", size = 12, alpha = 1, color = "r",bbox = dict(facecolor = "r", alpha = 0.1))

    # save fig
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = dirname + 'SVC-linear_ROC.png'
    pyplot.savefig(file)
    plt.show()


def calculate_metric(y_test, y_pre):
    confusion = confusion_matrix(y_test, y_pre, labels=None, sample_weight=None)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    print('Sensitivity:',TP / float(TP+FN))
    print('Specificity:',TN / float(TN+FP))
    Sensitivity=TP / float(TP+FN)
    Specificity=TN / float(TN+FP)
    return Sensitivity,Specificity


def save_proba_to_txt(test_proba, txtfile, label_name):
    f = open(txtfile, 'a')
    for j in range(0,len(test_proba)):
        proba = str(label_name[j]) + '\t' + str(test_proba[j]) + '\n'
        f.write(proba)
    f.close()


if __name__ == "__main__":
    Dirname = 'data/'
    filename = Dirname + 'TCGA_CHOL_PAAD_imData.txt'
    ID, feature, label, label_name, fea_name = load_data(filename)
    label = label.astype('int')

    # Get the best parameters of 5-fold cross validation output
    best_paras=[]
    with open(Dirname+"best_para_SVC_linear.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            best_paras.append(line)

    # Number of features
    feature_number = int(best_paras[2].split(':')[1])
    print("model:SVC_linear; feature_number:%d" %(feature_number))
    # dataset split
    X_train, X_test, y_train, y_test = train_test_split(feature, label_name, test_size=0.2, random_state=42)
    # feature selection
    X_train, selected_feature_index = select_feature(X_train, y_train, feature_number)
    X_test= load_some_Column(X_test, selected_feature_index)

    # ampullary adenocarcinoma dataset
    AACfile = Dirname + 'Zhejiang_AAC_imData.txt'
    AAC_ID, AAC_feature, AAC_label, AAC_label_name, AAC_fea_name = load_data(AACfile)
    AAC_feature_new= load_some_Column(AAC_feature, selected_feature_index)

    # External validation dataset
    validFile = Dirname + 'SYSUCC_CHOL_PAAD_imData.txt'
    val_ID, val_feature, val_label, val_label_name, val_fea_name = load_data(validFile)
    val_feature_new= load_some_Column(val_feature, selected_feature_index)

    print(Counter(label_name))

    # train model
    Model = train_model(X_train, y_train, C_para=float(best_paras[1].split(':')[1]))

    print('------------------test CHOL & PAAD data----------------------')
    test_Dir = Dirname+'test_results/'
    if not os.path.exists(test_Dir):
        os.makedirs(test_Dir)
    y_proba = test_model(Model, X_test, y_test, 'PAAD', savedir=test_Dir)
    y_proba_file = test_Dir + 'TCGA_test_proba_SVC_linear.txt'
    save_proba_to_txt(y_proba, y_proba_file, y_test)
    # plot ROC curve
    dirname = Dirname + 'test_results/ROC_Figures/'
    plot_ROC_curve(y_test, y_proba, dirname=dirname)

    # ampullary adenocarcinoma dataset test
    print('------------------test AAC data----------------------')
    AAC_Dir = Dirname+'zj_AAC_results/'
    if not os.path.exists(AAC_Dir):
        os.makedirs(AAC_Dir)
    AAC_proba = AAC_test_model(Model, AAC_feature_new)
    AAC_proba_file = AAC_Dir + 'AAC_proba_SVC_linear.txt'
    save_proba_to_txt(AAC_proba, AAC_proba_file, AAC_label_name)

    # external validation
    print('------------------external validation: CHOL&PAAD data----------------------')
    valid_Dir = Dirname+'zs_val_results/'
    if not os.path.exists(valid_Dir):
        os.makedirs(valid_Dir)
    val_proba = test_model(Model, val_feature_new, val_label_name, 'PAAD', savedir=valid_Dir)
    valid_proba_file = valid_Dir + 'valid_proba_SVC_linear.txt'
    save_proba_to_txt(val_proba, valid_proba_file, val_label_name)
    # plot ROC curve
    dirname = valid_Dir + 'ROC_Figures/'
    plot_ROC_curve(val_label_name, val_proba, dirname=dirname)
