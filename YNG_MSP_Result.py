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

#Load list of possible PSR
file = open('psr_candidates_top_3_model.txt','r')
test = file.readlines()
file.close()
i = 0
le = len(test)
save = []

while(i<le):
    a = test[i]
    proc_a = a[1:-2]
    save.append(proc_a)
    i += 1
    
#Load the FL8Y Catalog
my_dataframe = pd.read_csv("gll_psc_8year_v5_psc.csv", sep=",")

#Drop data that are missing value
my_dataframe = my_dataframe.dropna(subset=['Conf_95_SemiMajor','Signif_Avg'])

#Select unclassified data
my_dataframe = my_dataframe[my_dataframe.CLASS == '       ']

#Helper function for selecting possible PSR from catalog
def listcheck(name):
    for psr in save:
        if(name.startswith(psr)):
            return 1
    return -1

my_dataframe["psr_list"] = my_dataframe["Source_Name"].apply(lambda name: float(listcheck(name)))
my_dataframe = my_dataframe[my_dataframe.psr_list != -1]

#Features
my_dataframe["log_flux_density"] = np.log(my_dataframe["Flux_Density"])
my_dataframe["log_ROI_num"] = np.log(my_dataframe["ROI_num"])
my_dataframe["log_pivot_energy"] = np.log(my_dataframe["Pivot_Energy"])
my_dataframe["log_flux1000"] = np.log(my_dataframe["Flux1000"])
my_dataframe["log_energy_flux100"] = np.log(my_dataframe["Energy_Flux100"])
my_dataframe["log_npred"] = np.log(my_dataframe["Npred"])
my_dataframe["log_signif_avg"] = np.log(my_dataframe["Signif_Avg"])

def preprocess_features(my_dataframe):
    selected_features = my_dataframe[["log_flux_density","log_ROI_num","log_pivot_energy","log_flux1000","log_energy_flux100","LP_Index","LP_beta","log_npred","log_signif_avg","PLEC_Index","PLEC_Expfactor","GLAT"]]
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(my_dataframe):
    output_targets = my_dataframe[["is_msp"]]
    return output_targets

#Get the feature for data
test= preprocess_features(my_dataframe)

features = test.columns[:12]

#Name for the classifiers
names = ["KNeighborsClassifier", "LinearSVM", "GaussianProcessClassifier",
         "DecisionTreeClassifier", "RandomForestClassifier", "NeuralNet", "AdaBoost",
         "NaiveBayes", "QDA"]

#Dataframe for storing results
df = pd.DataFrame()
df["Source_Name"] = my_dataframe["Source_Name"]
    
#Predict results for each classfiier
for name in names:
    cl = pickle.load(open(name+'_MSP.sav','rb'))
    preds = cl.predict(test[features])
    df[name] = preds

#Results for 90%-up-model and top-3-model
df['90upmodel'] = df[["KNeighborsClassifier", "LinearSVM", "GaussianProcessClassifier",
         "DecisionTreeClassifier", "RandomForestClassifier", "NeuralNet", "AdaBoost",
         "NaiveBayes", "QDA"]].min(axis=1)
df['top3model'] = df[["KNeighborsClassifier","GaussianProcessClassifier","QDA"]].mode(axis=1)

#Convert 0,1 to YNG,MSP
def converter(num):
    if (num==0):
        return 'YNG'
    elif(num==1):
        return 'MSP'
    else:
        return num

df = df.applymap(converter)

#Save the results as csv
df.to_csv('result_msp.csv',sep='\t', index=False)

#Save two models results
df[df['90upmodel'] == 'MSP']["Source_Name"].to_csv('msp_candidates_90_up_model.txt', header=None, index=None, sep=' ')
df[df['top3model'] == 'MSP']["Source_Name"].to_csv('msp_candidates_top_3_model.txt', header=None, index=None, sep=' ')

print("Done!")
