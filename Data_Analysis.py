import pandas as pd 
import numpy as np 
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.preprocessing import OneHotEncoder


class EDA:

    def __init__(self, path,filename,sub_size = 2,remove_col= 'null',graph= False,dep_variable = 'null',num_cat= 'null'):
        self.path = path 
        self.filename = filename
        self.sub_size = sub_size
        self.remove_col = remove_col
        self.graph = graph
        self.dep_variable = dep_variable
        self.num_cat = num_cat


    def read_data(self):
        file = self.path + self.filename
        data = pd.read_csv(file)

        if self.remove_col == 'null':
            pass
        else:
            data.drop(columns=self.remove_col, axis = 1, inplace= True)
            
        return data
    
        
    
    def numeric_analyis(self):  
        '''
        Note : If remove col function is used it, the graph associated will that column will be removed, in case you wanna observe
               pass list 
                
        '''
        dep_var = pd.DataFrame()
        df = self.read_data()
        df = df.apply(pd.to_numeric , errors = 'coerce').dropna(axis=1)

        # For seperating dependent variable 
        if self.dep_variable == 'null':
            pass
        else:
            dep_var = df[self.dep_variable]
            df.drop(columns= self.dep_variable, axis=1,inplace=True)
        

        # Numeric Catagory
        if self.num_cat == 'null':
            pass
        else:
            df[self.num_cat +'_cat'] = [0 if i == 999 else 1 for i in df[self.num_cat]]
            df.drop([self.num_cat], inplace= True, axis=1)

        if self.graph:
            df.hist(column=df.columns, grid=False, figsize=(16,16), color = 'black')
            # Heat_Map 
            correlation = df.corr(method= "pearson")
            plt.figure(figsize=(25,10))
            sns.heatmap(correlation,vmax=1, square= True, annot= True)
            plt.show()

        return df, dep_var
        
    '''
    Furture work generalize this more especially textual analysis part

    '''    

    def textual_analysis(self):
        df = self.read_data()
        numeric = df.apply(pd.to_numeric , errors = 'coerce').dropna(axis=1)
        num_col = numeric.columns
        
        df = df[[col for col in df.columns if col not in num_col]]
        columns = df.columns


        # One hot encoding
        encoded_data = pd.get_dummies(df)
        
        if self.graph:
            fig, axes = plt.subplots(4, 3, figsize=(16, 16))
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.3)
            for i, ax in enumerate(axes.ravel()):
                if i >= len(columns):
                    ax.set_visible(False)
                    continue
                sns.countplot(y = columns[i], data=df, ax=ax, color='black')
            plt.show()
        return encoded_data
    
    def norm(self, data):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_norm = pd.DataFrame(min_max_scaler.fit_transform(data), columns= data.columns)
        return data_norm
    
    def Standard_scaler(self, data):
        scaler = StandardScaler()
        data_std = pd.DataFrame(scaler.fit_transform(data), columns= data.columns)
        return data_std


    def smote(self, x_train,y_train):
        smote =  SMOTE(sampling_strategy='minority',random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(x_train, y_train) 
        return X_train_smote,y_train_smote

    def resample(self,x_train, y_train):
        df = pd.concat([x_train,y_train],axis=1)

        print(df.shape)
        df_majority = df[df['term_deposit']==0] 
        df_minority = df[(df['term_deposit']==1)] 
        
        df_minority_upsampled = resample(df_minority, 
                                 replace=True,    
                                 n_samples= len(df_majority), 
                                 random_state=42)   
        df_upsampled = pd.concat([df_minority_upsampled, df_majority])
        #print(df_upsampled.shape)
        #final_df = pd.concat([df,df_upsampled], axis=0)
        #print(final_df.shape)

        df_test = df_upsampled['term_deposit']
        df_train = df_upsampled.drop(columns=['term_deposit'], axis=1)

        return df_train,df_test
    

