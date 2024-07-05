import pandas as pd
import numpy as np 
import Data_Analysis as Da
from sklearn import preprocessing
import model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score , precision_recall_curve , auc
import imblearn
from collections import Counter
from scipy.stats import randint


file_name = "European_bank_marketing.csv"
path ='D:\\Data_source_project\Homework\\Machine_learning\\Data\\'

obj = Da.EDA(path, file_name, graph=False, dep_variable=['term_deposit'],num_cat='pdays',remove_col=['duration'])

num,y = obj.numeric_analyis() 
txt = obj.textual_analysis()
txt = txt.astype(int)
df = pd.concat([num,txt],axis=1)

# Normalization of the data\\
#std_num = obj.Standard_scaler(num)

#data_norm = pd.concat([std_num,txt], axis=1)
data_norm = obj.norm(df)

thresholds = [0.1,0.2,0.35,0.5]

x_train, x_test, y_train, y_test = train_test_split(data_norm, y, test_size=0.3, random_state=42)
obj1 = model.Models(x_train,y_train)
pred_prob = obj1.algo(x_test , model_name = 'xg')
#obj1.cross_validation(thresholds = thresholds)
obj1.metrics(y_test,pred_prob,thresholds)
obj1.metric_plot_auc(y_test,pred_prob)
