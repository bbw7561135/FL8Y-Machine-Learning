#Import libraries (numpy,pandas,sklearn,pickle)
import numpy as np
import pandas as pd
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

#Seed for random separation of test data and training data
np.random.seed(0)

#Load the FL8Y Catalog
my_dataframe = pd.read_csv("gll_psc_8year_v5_psc.csv", sep=",")

#Drop data that are missing value or not AGN or not PSR
my_dataframe = my_dataframe[my_dataframe.CLASS != '       ']
my_dataframe = my_dataframe.dropna(subset=['Signif_Avg'])
my_dataframe = my_dataframe.reindex(np.random.permutation(my_dataframe.index))
my_dataframe["is_bll_bool"] = (my_dataframe["CLASS"].apply(lambda name: name.startswith("bll"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("BLL"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("FSRQ"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("fsrq"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("BCU"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("bcu"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("RDG"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("rdg"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("NLSY1"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("nlsy1"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("agn"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("ssrq"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("sey")))
my_dataframe["is_psr_bool"] = (my_dataframe["CLASS"].apply(lambda name: name.startswith("psr"))) | (my_dataframe["CLASS"].apply(lambda name: name.startswith("PSR")))
my_dataframe['bll_float'] = my_dataframe["is_bll_bool"].apply(lambda name: float(name))
my_dataframe['psr_float'] = my_dataframe["is_psr_bool"].apply(lambda name: float(name)*2)
my_dataframe["AGN_or_PSR"] = my_dataframe['bll_float'] + my_dataframe['psr_float']
my_dataframe = my_dataframe[my_dataframe.AGN_or_PSR != 0.0]
my_dataframe["AGN_or_PSR"] = my_dataframe["AGN_or_PSR"] - 1.0

#Features
my_dataframe["log_flux_density"] = np.log(my_dataframe["Flux_Density"])
my_dataframe["log_ROI_num"] = np.log(my_dataframe["ROI_num"])
my_dataframe["log_pivot_energy"] = np.log(my_dataframe["Pivot_Energy"])
my_dataframe["log_flux1000"] = np.log(my_dataframe["Flux1000"])
my_dataframe["log_energy_flux100"] = np.log(my_dataframe["Energy_Flux100"])
my_dataframe["log_npred"] = np.log(my_dataframe["Npred"])
my_dataframe["log_signif_avg"] = np.log(my_dataframe["Signif_Avg"])

def preprocess_features(my_dataframe):
    selected_features = my_dataframe[["log_flux_density","log_ROI_num","log_pivot_energy","log_flux1000","log_energy_flux100","LP_Index","LP_beta","log_npred","log_signif_avg","PLEC_Index","PLEC_Expfactor"]]
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(my_dataframe):
    output_targets = my_dataframe[["AGN_or_PSR"]]
    return output_targets

#Name and function for the classifiers
names = ["KNeighborsClassifier", "LinearSVM", "GaussianProcessClassifier",
         "DecisionTreeClassifier", "RandomForestClassifier", "NeuralNet", "AdaBoost",
         "NaiveBayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

#Output file name
outf = open("AGN_PSR_result.txt","w")

#Loop through different classifiers
for name, classifier in zip(names, classifiers):
    print(name)
    print(name, file=outf)
    max_acc = 0.0
    max_tpr = 0.0
    max_f1 = 0.0
    max_tnr = 0.0
    max_prec = 0.0

    #Training process (Repeated from the beginning for 1000 times)
    for loop in range(1000):
        #Separate 75% data to training data, 25% data to test data
        my_dataframe['is_train'] = np.random.uniform(0, 1, len(my_dataframe)) <= .75
        train, test = my_dataframe[my_dataframe['is_train']==True], my_dataframe[my_dataframe['is_train']==False]

        #Get the feature and target for training data and test data
        train_feature = preprocess_features(train)
        train_target = preprocess_targets(train)
        y = pd.factorize(train_target["AGN_or_PSR"])[0]

        test_feature= preprocess_features(test)
        test_target = preprocess_targets(test)

        features = train_feature.columns[:11]

        #Select classifier and train
        clf = classifier
        clf.fit(train_feature[features], y)

        #Predict the result for test data
        preds = clf.predict(test_feature[features])

        #Result
        print(loop)
        if(pd.crosstab(test_target["AGN_or_PSR"], preds, rownames=['Actual'], colnames=['Predicted']).size == 4):
            print(pd.crosstab(test_target["AGN_or_PSR"], preds, rownames=['Actual'], colnames=['Predicted']))
            tn = pd.crosstab(test_target["AGN_or_PSR"], preds, rownames=['Actual'], colnames=['Predicted']).iat[0,0]
            fp = pd.crosstab(test_target["AGN_or_PSR"], preds, rownames=['Actual'], colnames=['Predicted']).iat[0,1]
            fn = pd.crosstab(test_target["AGN_or_PSR"], preds, rownames=['Actual'], colnames=['Predicted']).iat[1,0]
            tp = pd.crosstab(test_target["AGN_or_PSR"], preds, rownames=['Actual'], colnames=['Predicted']).iat[1,1]

            #Calculate TPR, TNR, F1 score, Accuracy, Precision
            tpr = tp/(tp+fn)
            tnr = tn/(tn+fp)
            f1 = (2*tp)/(2*tp+fp+fn)
            acc = (tp+tn)/(tp+tn+fp+fn)
            prec = tp/(tp+fp)

            #Output result
            print("TPR", tpr, "TNR", tnr, "F1 score", f1, "Accuracy", acc, "Precision", prec)
            print()

        #Save the best result (highest accuracy)
        if(f1>max_f1):
            max_tpr = tpr
            max_tnr = tnr
            max_f1 = f1
            max_prec = prec
            max_acc = acc
            filename = name+'.sav'
            pickle.dump(clf,open(filename,'wb'))

    #Describe best model for this classifier
    print("Max accuracy model: TPR " + str(max_tpr) + " TNR " + str(max_tnr) + " F1 score " + str(max_f1) + " Accuracy " + str(max_acc) + " Precision " + str(max_prec))
    print("Max accuracy model: TPR " + str(max_tpr) + " TNR " + str(max_tnr) + " F1 score " + str(max_f1) + " Accuracy " + str(max_acc) + " Precision " + str(max_prec), file=outf)
    print("",file=outf)
    print()
    
outf.close()
