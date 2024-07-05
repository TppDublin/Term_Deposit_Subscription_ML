import pandas as pd
import numpy as np 
import Data_Analysis as Da


file_name = "European_bank_marketing.csv"
path ='D:\\Data_source_project\Homework\\Machine_learning\\Data\\'

obj = Da.EDA(path, file_name, graph= True, num_cat='pdays')

num,y = obj.numeric_analyis() 
data = obj.textual_analysis()

