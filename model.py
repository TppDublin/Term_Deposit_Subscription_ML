import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import Data_Analysis as Da
from sklearn.metrics import roc_curve, roc_auc_score , precision_recall_curve , auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

class Models:
    
    def __init__(self,features, benchmark):
        self.features = features 
        self.benchmark = benchmark

    def algo(self, x_test,model_name = 'lr', estimator = 40, hyper = False):

        dic = { 
                 'lr' : LogisticRegression(solver='liblinear', penalty='l1',random_state=42),
                 'rf': RandomForestClassifier(n_estimators= estimator),
                 'svm':  SVC(kernel='linear', probability= True),
                 'xg': GradientBoostingClassifier(),
                 'Knn': KNeighborsClassifier()
    
        }


        dic[model_name].fit(self.features,np.ravel(self.benchmark)) 
        y_pred_prob = dic[model_name].predict_proba(x_test)[:,1]


        if hyper:
            param_grid={#'max_depth':list(np.arange(10, 100, step=10)) + [None],
              'n_estimators':np.arange(10, 100, step=10),
              #'criterion':['gini','entropy'],
              #'min_samples_split':np.arange(2, 10, step=2)
            }  
            
            CV_rfc = GridSearchCV(estimator= dic[model_name], param_grid=param_grid, cv= 5, scoring='f1')
            CV_rfc.fit(self.features, self.benchmark)
            print(CV_rfc.best_params_)


        return y_pred_prob
    
    def kmean(self, feature_list):

        kmeans = KMeans(n_clusters=2,random_state=42, n_init=20)
        #'h_kmeans' :AgglomerativeClustering(distance_threshold=0,n_clusters=None,linkage='complete'
        data = self.features[feature_list]
        fit_data = kmeans.fit(data)
        data = pd.DataFrame(fit_data.labels_, columns='Kmean')
        return data 


    
    def cross_validation(self,thresholds,model_name = 'lr',k_type = 'norm', folds = 10):

        dic = { 'lr' : LogisticRegression(max_iter= 1000),
                 'rf': RandomForestClassifier(),
                 'svm':  SVC(gamma='scale'),
                 'xg': GradientBoostingClassifier()
        }

        k = { 'norm' : KFold(n_splits= folds, shuffle=True, random_state=1),
                    'stratified' : StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
                }

        #  1 : It represents PNRG algorithm which ensures that the split generated are same each time the code is run 
        for threshold in thresholds:

            accuracy_scores = []
            f1_scores = []
            precision_scores = []
            recall_scores = []
            
            for train_id, val_id in k[k_type].split(self.features,self.benchmark):
                X_train_fold, X_val_fold = self.features.iloc[train_id], self.features.iloc[val_id]
                y_train_fold, y_val_fold = self.benchmark.iloc[train_id], self.benchmark.iloc[val_id]

                dic[model_name].fit(X_train_fold,np.ravel(y_train_fold)) 
                prob = dic[model_name].predict_proba(X_val_fold)[:,1]
                
                # calculating score
                y_pred = (prob > threshold).astype(int)
                accuracy_scores.append(round(accuracy_score(y_val_fold, y_pred),3))
                f1_scores.append(round(f1_score(y_val_fold, y_pred), 3))
                precision_scores.append(round(precision_score(y_val_fold, y_pred),3))
                recall_scores.append(round(recall_score(y_val_fold, y_pred),3))

            print("----------------------------------------------------------------")
            print("Cross-Validification scores for Thresold value :", threshold) 
            print("----------------------------------------------------------------")       
            print("Accuracy(%) :",round(np.array(accuracy_scores).mean(),3))
            print("Precision(%) :",round(np.array(precision_scores).mean(),3))
            print("Recall(%) :", round(np.array(recall_scores).mean(),3))
            print("F1_score(%) :", round(np.array(f1_scores).mean(),3))


    def c_matrix(self,y_test, prediction):
        print("Confusion Matrix","\n")
         
        score = round(accuracy_score(np.ravel(y_test), prediction),3)
        f1 = round(f1_score(y_test, prediction), 3)
        precision = round(precision_score(y_test,prediction),3)
        recall = round(recall_score(y_test,prediction),3)

        print(classification_report(y_test,prediction))

        cm1= confusion_matrix(y_test, prediction)

        sns.heatmap(cm1, annot=True, fmt=".1f", linewidths=.3, 
                            square = True, cmap = 'PuBu')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title('Recall: {0}, Precision: {1}, F1-Score: {2}'.format(recall, precision, f1), size = 12)
        plt.show()
        print("\n")
        
        return cm1
    
    def metrics(self,y_test,prob,thresholds):
        tpr = []
        fpr = [] 
        precision = []
        auc = 0.0


        for threshold in thresholds:
            y_pred = (prob >= threshold).astype(int)
            cm1 = self.c_matrix(y_test, y_pred)
            tpr.append(cm1[1,1]/(cm1[1,1] + cm1[1,0]))
            fpr.append(cm1[0,1]/(cm1[0,1] + cm1[0,0]))
            precision.append(cm1[1,1]/(cm1[1,1] + cm1[0,1]))

        for i in range(1, len(tpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

        #print("Manual AUC for 4 thresold : ", auc)
    
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve with Different Thresholds')
        plt.grid(True)
        for i, threshold in enumerate(thresholds):
            plt.annotate(f'Threshold: {threshold}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(10,10), ha='center')

        plt.show()


        plt.plot(tpr, precision, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.grid(True)
        for i, threshold in enumerate(thresholds):
            plt.annotate(f'Threshold: {threshold}', (tpr[i], precision[i]), textcoords="offset points", xytext=(10,10), ha='center')

        plt.show()
    
    
    @staticmethod
    def metric_plot_auc(benchmark, prob):
    # no skill probs
        ns_prob = [0 for _ in range(len(benchmark))]
        no_skill = len(benchmark[benchmark == 1])/len(benchmark)

    # calculating the auc score 
        ns_auc = roc_auc_score(benchmark,ns_prob)
        lr_auc = roc_auc_score(benchmark,prob)

    # Calculate roc and precision-recall curve values

        ns_fpr, ns_tpr,_ = roc_curve(benchmark,ns_prob)
        lr_fpr, lr_tpr,_ = roc_curve(benchmark,prob)
        lr_precision, lr_recall, _= precision_recall_curve(benchmark,prob)
        auc_precision = auc(lr_recall, lr_precision)

        print('--------- Metrics -----------------')
        print('No skill: ROC AUC=%.3f'% (ns_auc))
        print('Logistic: ROC AUC=%.3f'% (lr_auc))
        print('Logistic : Precision-Recall AUC=%.3f'% (auc_precision))


        plt.plot(ns_fpr,ns_tpr,linestyle ='--' , label = 'No Skill')
        plt.plot(lr_fpr,lr_tpr, label = ' Logistic' )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Negative Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        plt.plot(lr_recall, lr_precision, label = 'Logistic')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve')
        plt.legend()
        plt.show()

    def PCA(self, data,desired_variance_ratio = 0.95):
        pca = PCA()
        pca.fit(data)

        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        optimal_n_components = np.argmax(cumulative_variance_ratio >= desired_variance_ratio) + 1
        print("Optimal number of components selected based on explained variance ratio:", optimal_n_components)
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()


        pca_f = PCA(n_components=optimal_n_components)
        x_train_pca = pca_f.fit_transform(self.features)

        return x_train_pca
    
