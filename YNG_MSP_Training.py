import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import pickle

pd.options.display.max_rows = 10
pd.options.display.max_columns = 40
pd.options.display.float_format = '{:.1f}'.format

file = open('psr_list.txt','r')
test = file.readlines()
file.close()
i = 0
le = len(test)
save = []

while(i<le):
    a = test[i]
    proc_a = a[:-1]
    b = test[i+1]
    proc_b = b[:-1]
    save.append((proc_a,proc_b))
    i += 2
    
my_dataframe = pd.read_csv("gll_psc_8year_v5_psc.csv", sep=",")

my_dataframe = my_dataframe.dropna(subset=['Conf_95_SemiMajor','Signif_Avg'])
my_dataframe = my_dataframe[my_dataframe.CLASS != '       ']
my_dataframe = my_dataframe.reindex(np.random.permutation(my_dataframe.index))
my_dataframe["is_bll_bool"] = (my_dataframe["CLASS"].apply(lambda name: name.startswith("bll"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("BLL"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("FSRQ"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("fsrq"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("BCU"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("bcu"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("RDG"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("rdg"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("NLSY1"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("nlsy1"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("agn"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("ssrq"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("sey")))
my_dataframe["is_psr_bool"] = (my_dataframe["CLASS"].apply(lambda name: name.startswith("psr"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("PSR")))
my_dataframe['bll_float'] = my_dataframe["is_bll_bool"].apply(lambda name: float(name))
my_dataframe['psr_float'] = my_dataframe["is_psr_bool"].apply(lambda name: float(name)*2)
my_dataframe["AGN_or_PSR"] = my_dataframe['bll_float'] + my_dataframe['psr_float']
my_dataframe = my_dataframe[my_dataframe.AGN_or_PSR != 0.0]
my_dataframe["AGN_or_PSR"] = my_dataframe["AGN_or_PSR"] - 1.0

#my_dataframe = my_dataframe[my_dataframe.is_psr_bool != 0.0]

def listcheck(name):
    for (psr,m) in save:
        if(name.startswith(psr)):
            return m
    return -1

my_dataframe["is_msp"] = my_dataframe["ASSOC1"].apply(lambda name: float(listcheck(name)))
my_dataframe = my_dataframe[my_dataframe.is_msp != -1]

my_dataframe["log_flux_density"] = np.log(my_dataframe["Flux_Density"])

my_dataframe["log_ROI_num"] = np.log(my_dataframe["ROI_num"])

my_dataframe["log_pivot_energy"] = np.log(my_dataframe["Pivot_Energy"])

my_dataframe["log_flux1000"] = np.log(my_dataframe["Flux1000"])

my_dataframe["log_energy_flux100"] = np.log(my_dataframe["Energy_Flux100"])

my_dataframe["log_npred"] = np.log(my_dataframe["Npred"])

my_dataframe["log_signif_avg"] = np.log(my_dataframe["Signif_Avg"])
my_dataframe["log_Conf_95_SemiMajor"] = np.log(my_dataframe["Conf_95_SemiMajor"])
my_dataframe["log_Conf_95_SemiMinor"] = np.log(my_dataframe["Conf_95_SemiMinor"])

np.random.seed(0)

def preprocess_features(my_dataframe):
    selected_features = my_dataframe[["log_flux_density","log_ROI_num","log_pivot_energy","log_flux1000","log_energy_flux100","LP_Index","LP_beta","log_npred","log_signif_avg","PLEC_Index","PLEC_Expfactor","GLAT"]]
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(my_dataframe):
    output_targets = my_dataframe[["is_msp"]]
    return output_targets

names = ["KNeighborsClassifier", "LinearSVM", "RBFSVM", "GaussianProcessClassifier",
         "DecisionTreeClassifier", "RandomForestClassifier", "NeuralNet", "AdaBoost",
         "NaiveBayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

outf = open("YNG_MSP_result.txt","w")

for name, classifier in zip(names, classifiers):
    print(name)
    print(name, file=outf)
    max_acc = 0.0
    max_tpr = 0.0
    max_f1 = 0.0
    max_tnr = 0.0
    max_prec = 0.0
    for loop in range(1000):
        my_dataframe['is_train'] = np.random.uniform(0, 1, len(my_dataframe)) <= .7
        train, test = my_dataframe[my_dataframe['is_train']==True], my_dataframe[my_dataframe['is_train']==False]

        train_feature = preprocess_features(train)
        train_target = preprocess_targets(train)
        y = pd.factorize(train_target["is_msp"])[0]

        test_feature= preprocess_features(test)
        test_target = preprocess_targets(test)

        features = train_feature.columns[:12]

        #choose classifier
        clf = classifier
        clf.fit(train_feature[features], y)

        #predict
        preds = clf.predict(test_feature[features])

        #result
        print(loop)
        print(pd.crosstab(test_target["is_msp"], preds, rownames=['Actual'], colnames=['Predicted']))
        if(pd.crosstab(test_target["is_msp"], preds, rownames=['Actual'], colnames=['Predicted']).size == 4):
            tn = pd.crosstab(test_target["is_msp"], preds, rownames=['Actual'], colnames=['Predicted']).iat[0,0]
            fp = pd.crosstab(test_target["is_msp"], preds, rownames=['Actual'], colnames=['Predicted']).iat[0,1]
            fn = pd.crosstab(test_target["is_msp"], preds, rownames=['Actual'], colnames=['Predicted']).iat[1,0]
            tp = pd.crosstab(test_target["is_msp"], preds, rownames=['Actual'], colnames=['Predicted']).iat[1,1]

            tpr = tp/(tp+fn)
            tnr = tn/(tn+fp)
            f1 = (2*tp)/(2*tp+fp+fn)
            acc = (tp+tn)/(tp+tn+fp+fn)
            prec = tp/(tp+fp)

            #describe
            print("TPR", tpr, "TNR", tnr, "F1 score", f1, "Accuracy", acc, "Precision", prec)
            print()
        if(acc>max_acc):
            max_acc = acc
            max_tpr = tpr
            max_tnr = tnr
            max_f1 = f1
            max_prec = prec
            filename = name+'_MSP.sav'
            pickle.dump(clf,open(filename,'wb'))
    print("Max accuracy model: TPR " + str(max_tpr) + " TNR " + str(max_tnr) + " F1 score " + str(max_f1) + " Accuracy " + str(max_acc) + " Precision " + str(max_prec))
    print("Max accuracy model: TPR " + str(max_tpr) + " TNR " + str(max_tnr) + " F1 score " + str(max_f1) + " Accuracy " + str(max_acc) + " Precision " + str(max_prec), file=outf)
    print("",file=outf)
    print()
    
outf.close()

