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

#Load the FL8Y Catalog
my_dataframe = pd.read_csv("gll_psc_8year_v5_psc.csv", sep=",")

#Drop data that are missing value
my_dataframe = my_dataframe.dropna(subset=['Conf_95_SemiMajor','Signif_Avg'])
my_dataframe = my_dataframe[my_dataframe.CLASS == '       ']

#Features
def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x:(x - mean) / std_dv)

my_dataframe["log_flux_density"] = np.log(my_dataframe["Flux_Density"])
my_dataframe["log_ROI_num"] = np.log(my_dataframe["ROI_num"])
my_dataframe["log_pivot_energy"] = np.log(my_dataframe["Pivot_Energy"])
my_dataframe["log_flux1000"] = np.log(my_dataframe["Flux1000"])
my_dataframe["log_energy_flux100"] = np.log(my_dataframe["Energy_Flux100"])
my_dataframe["log_npred"] = np.log(my_dataframe["Npred"])
my_dataframe["log_signif_avg"] = np.log(my_dataframe["Signif_Avg"])
my_dataframe["log_Conf_95_SemiMajor"] = np.log(my_dataframe["Conf_95_SemiMajor"])
my_dataframe["log_Conf_95_SemiMinor"] = np.log(my_dataframe["Conf_95_SemiMinor"])

def preprocess_features(my_dataframe):
    selected_features = my_dataframe[["log_flux_density","log_ROI_num","log_pivot_energy","log_flux1000","log_energy_flux100","LP_Index","LP_beta","log_npred","log_signif_avg","PLEC_Index","PLEC_Expfactor"]]
    processed_features = selected_features.copy()
    return processed_features

#Get the feature for data
test= preprocess_features(my_dataframe)

features = test.columns[:11]

#Name for the classifiers
names = ["KNeighborsClassifier", "LinearSVM", "RBFSVM", "GaussianProcessClassifier",
         "DecisionTreeClassifier", "RandomForestClassifier", "NeuralNet", "AdaBoost",
         "NaiveBayes", "QDA"]

#Dataframe for storing results
df = pd.DataFrame()
df["Source_Name"] = my_dataframe["Source_Name"]

#Predict results for each classfiier
for name in names:
    cl = pickle.load(open(name+'.sav','rb'))
    preds = cl.predict(test[features])
    df[name] = preds
    
#Save the results as csv
df.to_csv('result.csv',sep='\t', index=False)

print("Done!")
