# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:34:48 2019

@author: Arnold Yu
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------------------------------------------

# calculate accuracy    
def calculateAcc(matrix):
    denom = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1]
    num = matrix[0][0] + matrix[1][1]
    return float(num)/float(denom)
    
# calculate percision  
def calculatePer( matrix):
    denom = matrix[0][0]+ matrix[1][0]
    return float(matrix[0][0])/float(denom)
    
# calculate recall
def calculateRec( matrix):
    denom = matrix[0][0]+ matrix[0][1]
    return float(matrix[0][0])/float(denom)
    
# calculate F1
def calculateF1(percision, recall):
    return float(2.0 * percision * recall)/ float(percision + recall)

# calculate false positive rate
def FPR(matrix):
    denom = matrix[0][1]+ matrix[1][1]
    return float(matrix[0,1])/float(denom)

#----------------------------------------------------------------------------------------------------------------------------------------------------------

# import dataset
train = pd.read_csv('data/train_replaceMissing.csv')
test = pd.read_csv('data/test_replaceMissing.csv')

from sklearn.preprocessing import StandardScaler, LabelEncoder

# turn categorical data into numerical
le = LabelEncoder()
columns_cat = ['Workclass', 'Education', 'Marital', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country', 'Income']
for col in columns_cat:
    if train[col].dtype == object:
        train[col] = le.fit_transform(train[col])
        test[col] = le.fit_transform(test[col])
        
# standardlize      
sc = StandardScaler()
all_data = pd.concat([train, test])
all_X = all_data.iloc[:, :-1]
all_y = all_data.iloc[:, -1]
all_X = sc.fit_transform(all_X)

# import all classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


models = {'Naïve Bayesian': GaussianNB(),
          'Decision tree classifier': DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
          'k-NN': KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
          'Random Forest': RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
          'Logistic regression': LogisticRegression( solver='lbfgs', random_state = 0),
          'AdaBoost': AdaBoostClassifier(n_estimators = 50, learning_rate = 1. ,  random_state = 0),
          'SVM linear': SVC(kernel = 'linear', random_state = 0, probability = True) 
          }
results = {'Naïve Bayesian': {},
          'Decision tree classifier':  {},
          'k-NN': {},
          'Random Forest':  {},
          'Logistic regression':  {},
          'AdaBoost':  {},
          'SVM linear':  {} 
          }    
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
methodP = ['predict', 'predict_proba']
target_names = ['Class 0', 'Class 1']


#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Naïve Bayesian classifier

y_prob_nb = cross_val_predict(estimator = models['Naïve Bayesian'], 
                           X = all_X, 
                           y= all_y,
                           groups=None, 
                           cv= 10, 
                           n_jobs= -1, 
                           verbose=0, 
                           fit_params=None, 
                           pre_dispatch='2*n_jobs', 
                           method= methodP[1])

y_pred_nb = np.zeros(len(y_prob_nb))
for i in range(0, len(y_prob_nb)):
    if y_prob_nb[i][0] > y_prob_nb[i][1]:
        y_pred_nb[i] = 0
    else:
        y_pred_nb[i] = 1
#import sklearn
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
cm_nb = confusion_matrix(all_y, y_pred_nb)
dict_nb = classification_report(all_y, y_pred_nb, target_names = target_names,  output_dict= True)


results['Naïve Bayesian']['Confusion_matrix'] = cm_nb
results['Naïve Bayesian']['Classification_report'] = dict_nb
results['Naïve Bayesian']['Accuracy'] = calculateAcc(cm_nb)
results['Naïve Bayesian']['Percision'] = calculatePer(cm_nb)
results['Naïve Bayesian']['Recall'] = calculateRec(cm_nb)
results['Naïve Bayesian']['F1'] = calculateF1(results['Naïve Bayesian']['Percision'],results['Naïve Bayesian']['Recall'])

fpr, tpr, thresholds = roc_curve(all_y, y_prob_nb[:, 1])
auc = roc_auc_score(all_y, y_prob_nb[:, 1])
plt.plot(fpr, tpr, label='ROC curve:(area = %0.3f)' % (auc))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'Naïve Bayesian')
plt.legend(loc="lower right")


#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Decision tree classifier 
y_prob_dt = cross_val_predict(estimator = models['Decision tree classifier'], 
                           X = all_X, 
                           y= all_y,
                           groups=None, 
                           cv= 10, 
                           n_jobs=-1, 
                           verbose=0, 
                           fit_params=None, 
                           pre_dispatch='2*n_jobs', 
                           method= methodP[1])

y_pred_dt = np.zeros(len(y_prob_dt))
for i in range(0, len(y_prob_dt)):
    if y_prob_dt[i][0] > y_prob_dt[i][1]:
        y_pred_dt[i] = 0
    else:
        y_pred_dt[i] = 1
        
cm_dt = confusion_matrix(all_y, y_pred_dt)
dict_dt = classification_report(all_y, y_pred_dt, target_names = target_names,  output_dict= True)


results['Decision tree classifier']['Confusion_matrix'] = cm_dt
results['Decision tree classifier']['Classification_report'] = dict_dt
results['Decision tree classifier']['Accuracy'] = calculateAcc(cm_dt)
results['Decision tree classifier']['Percision'] = calculatePer(cm_dt)
results['Decision tree classifier']['Recall'] = calculateRec(cm_dt)
results['Decision tree classifier']['F1'] = calculateF1(results['Decision tree classifier']['Percision'],results['Decision tree classifier']['Recall'])

fpr, tpr, thresholds = roc_curve(all_y, y_prob_dt[:, 1])
auc = roc_auc_score(all_y, y_prob_dt[:, 1])
plt.plot(fpr, tpr, label='ROC curve:(area = %0.3f)' % (auc))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'Decision tree classifier')
plt.legend(loc="lower right")



#----------------------------------------------------------------------------------------------------------------------------------------------------------


# k-NN classifier 
y_prob_knn = cross_val_predict(estimator = models['k-NN'], 
                           X = all_X, 
                           y= all_y,
                           groups=None, 
                           cv= 10, 
                           n_jobs=-1, 
                           verbose=0, 
                           fit_params=None, 
                           pre_dispatch='2*n_jobs', 
                           method= methodP[1])

y_pred_knn = np.zeros(len(y_prob_knn))
for i in range(0, len(y_prob_knn)):
    if y_prob_knn[i][0] > y_prob_knn[i][1]:
        y_pred_knn[i] = 0
    else:
        y_pred_knn[i] = 1

cm_knn = confusion_matrix(all_y, y_pred_knn)
dict_knn = classification_report(all_y, y_pred_knn, target_names = target_names,  output_dict= True)


results['k-NN']['Confusion_matrix'] = cm_knn
results['k-NN']['Classification_report'] = dict_knn
results['k-NN']['Accuracy'] = calculateAcc(cm_knn)
results['k-NN']['Percision'] = calculatePer(cm_knn)
results['k-NN']['Recall'] = calculateRec(cm_knn)
results['k-NN']['F1'] = calculateF1(results['k-NN']['Percision'],results['k-NN']['Recall'])

fpr, tpr, thresholds = roc_curve(all_y, y_prob_knn[:, 1])
auc = roc_auc_score(all_y, y_prob_knn[:, 1])
plt.plot(fpr, tpr, label='ROC curve:(area = %0.3f)' % (auc))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'k-NN')
plt.legend(loc="lower right")




#----------------------------------------------------------------------------------------------------------------------------------------------------------




# Random Forest classifier 
y_prob_rf= cross_val_predict(estimator = models['Random Forest'], 
                           X = all_X, 
                           y= all_y,
                           groups=None, 
                           cv= 10, 
                           n_jobs= -1, 
                           verbose=0, 
                           fit_params=None, 
                           pre_dispatch='2*n_jobs', 
                           method= methodP[1])

y_pred_rf = np.zeros(len(y_prob_rf))
for i in range(0, len(y_prob_rf)):
    if y_prob_rf[i][0] > y_prob_rf[i][1]:
        y_pred_rf[i] = 0
    else:
        y_pred_rf[i] = 1

cm_rf = confusion_matrix(all_y, y_pred_rf)
dict_rf = classification_report(all_y, y_pred_rf, target_names = target_names,  output_dict= True)


results['Random Forest']['Confusion_matrix'] = cm_rf
results['Random Forest']['Classification_report'] = dict_rf
results['Random Forest']['Accuracy'] = calculateAcc(cm_rf)
results['Random Forest']['Percision'] = calculatePer(cm_rf)
results['Random Forest']['Recall'] = calculateRec(cm_rf)
results['Random Forest']['F1'] = calculateF1(results['Random Forest']['Percision'],results['Random Forest']['Recall'])

fpr, tpr, thresholds = roc_curve(all_y, y_prob_rf[:, 1])
auc = roc_auc_score(all_y, y_prob_rf[:, 1])
plt.plot(fpr, tpr, label='ROC curve:(area = %0.3f)' % (auc))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'Random Forest')
plt.legend(loc="lower right")



#----------------------------------------------------------------------------------------------------------------------------------------------------------


# Logistic regression classifier 
y_prob_logis = cross_val_predict(estimator = models['Logistic regression'], 
                           X = all_X, 
                           y= all_y,
                           groups=None, 
                           cv= 10, 
                           n_jobs= -1, 
                           verbose=0, 
                           fit_params=None, 
                           pre_dispatch='2*n_jobs', 
                           method= methodP[1])

y_pred_logis = np.zeros(len(y_prob_logis))
for i in range(0, len(y_prob_logis)):
    if y_prob_logis[i][0] > y_prob_logis[i][1]:
        y_pred_logis[i] = 0
    else:
        y_pred_logis[i] = 1

cm_logis = confusion_matrix(all_y, y_pred_logis)
dict_logis = classification_report(all_y, y_pred_logis, target_names = target_names,  output_dict= True)


results['Logistic regression']['Confusion_matrix'] = cm_logis
results['Logistic regression']['Classification_report'] = dict_logis
results['Logistic regression']['Accuracy'] = calculateAcc(cm_logis)
results['Logistic regression']['Percision'] = calculatePer(cm_logis)
results['Logistic regression']['Recall'] = calculateRec(cm_logis)
results['Logistic regression']['F1'] = calculateF1(results['Logistic regression']['Percision'],results['Logistic regression']['Recall'])

fpr, tpr, thresholds = roc_curve(all_y, y_prob_logis[:, 1])
auc = roc_auc_score(all_y, y_prob_logis[:, 1])
plt.plot(fpr, tpr, label='ROC curve:(area = %0.3f)' % (auc))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'Logistic regression')
plt.legend(loc="lower right")




#----------------------------------------------------------------------------------------------------------------------------------------------------------



# AdaBoost classifier 
y_prob_ada = cross_val_predict(estimator = models['AdaBoost'], 
                           X = all_X, 
                           y= all_y,
                           groups=None, 
                           cv= 10, 
                           n_jobs= -1, 
                           verbose=0, 
                           fit_params=None, 
                           pre_dispatch='2*n_jobs', 
                           method= methodP[1])

y_pred_ada = np.zeros(len(y_prob_ada))
for i in range(0, len(y_prob_ada)):
    if y_prob_ada[i][0] > y_prob_ada[i][1]:
        y_pred_ada[i] = 0
    else:
        y_pred_ada[i] = 1

cm_ada = confusion_matrix(all_y, y_pred_ada)
dict_ada = classification_report(all_y, y_pred_ada, target_names = target_names,  output_dict= True)


results['AdaBoost']['Confusion_matrix'] = cm_ada
results['AdaBoost']['Classification_report'] = dict_ada
results['AdaBoost']['Accuracy'] = calculateAcc(cm_ada)
results['AdaBoost']['Percision'] = calculatePer(cm_ada)
results['AdaBoost']['Recall'] = calculateRec(cm_ada)
results['AdaBoost']['F1'] = calculateF1(results['AdaBoost']['Percision'],results['AdaBoost']['Recall'])

fpr, tpr, thresholds = roc_curve(all_y, y_prob_ada[:, 1])
auc = roc_auc_score(all_y, y_prob_ada[:, 1])
plt.plot(fpr, tpr, label='ROC curve:(area = %0.3f)' % (auc))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'AdaBoost')
plt.legend(loc="lower right")




#----------------------------------------------------------------------------------------------------------------------------------------------------------





# svm linear classifier 
y_prob_svm = cross_val_predict(estimator = models['SVM linear'], 
                           X = all_X, 
                           y= all_y,
                           groups=None, 
                           cv= 10, 
                           n_jobs= -1, 
                           verbose=0, 
                           fit_params=None, 
                           pre_dispatch='2*n_jobs', 
                           method= methodP[1])

y_pred_svm = np.zeros(len(y_prob_svm))
for i in range(0, len(y_prob_svm)):
    if y_prob_svm[i][0] > y_prob_svm[i][1]:
        y_pred_svm[i] = 0
    else:
        y_pred_svm[i] = 1

cm_svm= confusion_matrix(all_y, y_pred_svm)
dict_svm = classification_report(all_y, y_pred_svm, target_names = target_names,  output_dict= True)


results['SVM linear']['Confusion_matrix'] = cm_svm
results['SVM linear']['Classification_report'] = dict_svm
results['SVM linear']['Accuracy'] = calculateAcc(cm_svm)
results['SVM linear']['Percision'] = calculatePer(cm_svm)
results['SVM linear']['Recall'] = calculateRec(cm_svm)
results['SVM linear']['F1'] = calculateF1(results['SVM linear']['Percision'],results['SVM linear']['Recall'])

fpr, tpr, thresholds = roc_curve(all_y, y_prob_svm[:, 1])
auc = roc_auc_score(all_y, y_prob_svm[:, 1])
plt.plot(fpr, tpr, label='ROC curve:(area = %0.3f)' % (auc))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'SVM linear')
plt.legend(loc="lower right")



#----------------------------------------------------------------------------------------------------------------------------------------------------------
# graph learning curves of all classifiers

from sklearn.model_selection import learning_curve


train_sizes, train_scores, test_scores = learning_curve(estimator = models['Naïve Bayesian'],
                                                        X = all_X, 
                                                        y = all_y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        # Evaluation metric
                                                        scoring='neg_log_loss', # this scoring can change to accuracy
                                                        # Use all computer cores
                                                        n_jobs = -1, 
                                                        # 10 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 10))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.grid()
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve : Naïve Bayesian")
plt.xlabel("Training Set Size"), plt.ylabel("Neg Log Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.show()





#----------------------------------------------------------------------------------------------------------------------------------------------------------




train_sizes, train_scores, test_scores = learning_curve(estimator = models['Decision tree classifier'],
                                                        X = all_X, 
                                                        y = all_y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        # Evaluation metric
                                                        scoring='neg_log_loss', # this scoring can change to accuracy
                                                        # Use all computer cores
                                                        n_jobs = -1, 
                                                        # 10 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 10))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.grid()
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve : Decision tree classifier")
plt.xlabel("Training Set Size"), plt.ylabel("Neg Log Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------


train_sizes, train_scores, test_scores = learning_curve(estimator = models['k-NN'],
                                                        X = all_X, 
                                                        y = all_y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        # Evaluation metric
                                                        scoring='neg_log_loss', # this scoring can change to accuracy
                                                        # Use all computer cores
                                                        n_jobs = -1, 
                                                        # 10 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 10))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.grid()
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve : k-NN")
plt.xlabel("Training Set Size"), plt.ylabel("Neg Log Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------



train_sizes, train_scores, test_scores = learning_curve(estimator = models['Random Forest'],
                                                        X = all_X, 
                                                        y = all_y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        # Evaluation metric
                                                        scoring='neg_log_loss', # this scoring can change to accuracy
                                                        # Use all computer cores
                                                        n_jobs = -1, 
                                                        # 10 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 10))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.grid()
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve : Random Forest")
plt.xlabel("Training Set Size"), plt.ylabel("Neg Log Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------



train_sizes, train_scores, test_scores = learning_curve(estimator = models['Logistic regression'],
                                                        X = all_X, 
                                                        y = all_y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        # Evaluation metric
                                                        scoring='neg_log_loss', # this scoring can change to accuracy
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 10 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 10))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.grid()
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve : Logistic regression")
plt.xlabel("Training Set Size"), plt.ylabel("Neg Log Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------




train_sizes, train_scores, test_scores = learning_curve(estimator = models['AdaBoost'],
                                                        X = all_X, 
                                                        y = all_y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        # Evaluation metric
                                                        scoring='neg_log_loss', # this scoring can change to accuracy
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 10 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 10))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.grid()
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve : AdaBoost")
plt.xlabel("Training Set Size"), plt.ylabel("Neg Log Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------



train_sizes, train_scores, test_scores = learning_curve(estimator = models['SVM linear'],
                                                        X = all_X, 
                                                        y = all_y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        # Evaluation metric
                                                        scoring='neg_log_loss', # this scoring can change to accuracy
                                                        # Use all computer cores
                                                        n_jobs = -1, 
                                                        # 10 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 10))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.grid()
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve : SVM linear")
plt.xlabel("Training Set Size"), plt.ylabel("Neg Log Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.show()



