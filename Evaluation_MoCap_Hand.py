# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 00:53:23 2019

@author: Arnold Yu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




data = pd.read_csv('data/allUser_preprocessed.csv')
X = data.iloc[:, 1:]
all_y = data.iloc[:, 0]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
all_X = sc.fit_transform(X)


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
          'Logistic regression': LogisticRegression( solver='lbfgs', max_iter = 10000, random_state = 0, multi_class = 'multinomial'),
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
from sklearn.metrics import auc
from scipy import interp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
#from sklearn.metrics import precision_recall_fscore_support
methodP = ['predict', 'predict_proba']
target_names = ['Class 1', 'Class 2', 'Class 3','Class 4','Class 5']

ohe = OneHotEncoder(categories = 'auto')
y = pd.DataFrame(all_y)
y = ohe.fit_transform(y).toarray()

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
    y_pred_nb[i] = y_prob_nb[i].argmax(axis = 0) + 1
    
cm_nb = confusion_matrix(all_y, y_pred_nb)
dict_nb = classification_report(all_y, y_pred_nb, target_names = target_names,  output_dict= True)


results['Naïve Bayesian']['Confusion_matrix'] = cm_nb
results['Naïve Bayesian']['Accuracy'] = accuracy_score(all_y, y_pred_nb)
results['Naïve Bayesian']['Classification_report'] = dict_nb


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(target_names)):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_prob_nb[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(),  y_prob_nb.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(target_names))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(target_names)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(target_names)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve:(area = %0.3f)' % (roc_auc["micro"]))
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve:(area = %0.3f)' % (roc_auc["macro"]))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'Naïve Bayesian')
plt.legend(loc="lower right")
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Decision tree classifier

y_prob_dt = cross_val_predict(estimator = models['Decision tree classifier'], 
                           X = all_X, 
                           y= all_y,
                           groups=None, 
                           cv= 10, 
                           n_jobs= -1, 
                           verbose=0, 
                           fit_params=None, 
                           pre_dispatch='2*n_jobs', 
                           method= methodP[1])

y_pred_dt = np.zeros(len(y_prob_dt))
for i in range(0, len(y_prob_dt)):
    y_pred_dt[i] = y_prob_dt[i].argmax(axis = 0) + 1
    
cm_dt = confusion_matrix(all_y, y_pred_dt)
dict_dt = classification_report(all_y, y_pred_dt, target_names = target_names,  output_dict= True)


results['Decision tree classifier']['Confusion_matrix'] = cm_dt
results['Decision tree classifier']['Accuracy'] = accuracy_score(all_y, y_pred_dt)
results['Decision tree classifier']['Classification_report'] = dict_dt



fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(target_names)):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_prob_dt[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(),  y_prob_dt.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(target_names))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(target_names)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(target_names)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve:(area = %0.3f)' % (roc_auc["micro"]))
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve:(area = %0.3f)' % (roc_auc["macro"]))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'Decision tree classifier')
plt.legend(loc="lower right")
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# k-NN classifier

y_prob_knn = cross_val_predict(estimator = models['k-NN'], 
                           X = all_X, 
                           y= all_y,
                           groups=None, 
                           cv= 10, 
                           n_jobs= -1, 
                           verbose=0, 
                           fit_params=None, 
                           pre_dispatch='2*n_jobs', 
                           method= methodP[1])

y_pred_knn = np.zeros(len(y_prob_knn))
for i in range(0, len(y_prob_knn)):
    y_pred_knn[i] = y_prob_knn[i].argmax(axis = 0) + 1
    
cm_knn = confusion_matrix(all_y, y_pred_knn)
dict_knn = classification_report(all_y, y_pred_knn, target_names = target_names,  output_dict= True)


results['k-NN']['Confusion_matrix'] = cm_knn
results['k-NN']['Accuracy'] = accuracy_score(all_y, y_pred_knn)
results['k-NN']['Classification_report'] = dict_knn


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(target_names)):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_prob_knn[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(),  y_prob_knn.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(target_names))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(target_names)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(target_names)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve:(area = %0.3f)' % (roc_auc["micro"]))
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve:(area = %0.3f)' % (roc_auc["macro"]))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'k-NN')
plt.legend(loc="lower right")
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Random Forest classifier

y_prob_rf = cross_val_predict(estimator = models['Random Forest'], 
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
    y_pred_rf[i] = y_prob_rf[i].argmax(axis = 0) + 1
    
cm_rf = confusion_matrix(all_y, y_pred_rf)
dict_rf = classification_report(all_y, y_pred_rf, target_names = target_names,  output_dict= True)


results['Random Forest']['Confusion_matrix'] = cm_rf
results['Random Forest']['Accuracy'] = accuracy_score(all_y, y_pred_rf)
results['Random Forest']['Classification_report'] = dict_rf


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(target_names)):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_prob_rf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(),  y_prob_rf.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(target_names))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(target_names)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(target_names)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve:(area = %0.3f)' % (roc_auc["micro"]))
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve:(area = %0.3f)' % (roc_auc["macro"]))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'Random Forest')
plt.legend(loc="lower right")
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Logistic regression classifier

"""from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
ovr = OneVsRestClassifier(estimator = models['Logistic regression'])

kf = KFold(n_splits = 10, shuffle = False, random_state = None)

y_prob_logis = np.zeros_like(y)

for train_index, test_index in kf.split(all_X): 
    # Split the data for 10-fold
    X_train, X_test = all_X[train_index], all_X[test_index]
    y_train, y_test = all_y[train_index], all_y[test_index]
        
    y_train = y_train[:,-1]
    y_test = y_test[:,-1]
    # Fit the training data
    ovr.fit(X_train, y_train)
    # Predicte the probability of the test result
    y_prob = ovr.predict_proba(X_test)
    y_prob_logis += y_prob
"""
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
    y_pred_logis[i] = y_prob_logis[i].argmax(axis = 0) + 1
    
cm_logis = confusion_matrix(all_y, y_pred_logis)
dict_logis = classification_report(all_y, y_pred_logis, target_names = target_names,  output_dict= True)


results['Logistic regression']['Confusion_matrix'] = cm_logis
results['Logistic regression']['Accuracy'] = accuracy_score(all_y, y_pred_logis)
results['Logistic regression']['Classification_report'] = dict_logis

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(target_names)):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_prob_logis[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(),  y_prob_logis.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(target_names))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(target_names)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(target_names)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve:(area = %0.3f)' % (roc_auc["micro"]))
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve:(area = %0.3f)' % (roc_auc["macro"]))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'Logistic regression')
plt.legend(loc="lower right")
plt.show()
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
    y_pred_ada[i] = y_prob_ada[i].argmax(axis = 0) + 1
    
cm_ada = confusion_matrix(all_y, y_pred_ada)
dict_ada = classification_report(all_y, y_pred_ada, target_names = target_names,  output_dict= True)


results['AdaBoost']['Confusion_matrix'] = cm_ada
results['AdaBoost']['Accuracy'] = accuracy_score(all_y, y_pred_ada)
results['AdaBoost']['Classification_report'] = dict_ada


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(target_names)):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_prob_ada[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(),  y_prob_ada.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(target_names))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(target_names)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(target_names)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve:(area = %0.3f)' % (roc_auc["micro"]))
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve:(area = %0.3f)' % (roc_auc["macro"]))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'AdaBoost')
plt.legend(loc="lower right")
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# SVM linear classifier

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
    y_pred_svm[i] = y_prob_svm[i].argmax(axis = 0) + 1
    
cm_svm = confusion_matrix(all_y, y_pred_svm)
dict_svm = classification_report(all_y, y_pred_svm, target_names = target_names,  output_dict= True)


results['SVM linear']['Confusion_matrix'] = cm_svm
results['SVM linear']['Accuracy'] = accuracy_score(all_y, y_pred_svm)
results['SVM linear']['Classification_report'] = dict_svm


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(target_names)):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_prob_svm[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(),  y_prob_svm.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(target_names))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(target_names)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(target_names)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve:(area = %0.3f)' % (roc_auc["micro"]))
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve:(area = %0.3f)' % (roc_auc["macro"]))
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or (Recall)')
plt.title('ROC : %s' % 'SVM linear')
plt.legend(loc="lower right")
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------
# graph learning curves of all classifiers

from sklearn.model_selection import learning_curve


train_sizes, train_scores, test_scores = learning_curve(estimator = models['Naïve Bayesian'],
                                                        X = all_X, 
                                                        y = all_y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        shuffle = True,
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
                                                        shuffle = True,
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
                                                        shuffle = True,
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
                                                        shuffle = True,
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
                                                        shuffle = True,
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
                                                        shuffle = True,
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
                                                        shuffle = True,
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

