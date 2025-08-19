import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tensorflow.keras import optimizers
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

df = pd.read_csv('path...') 
Data = df[['Cor', 'TTP', 'MTT_SVD',  'PH', 'AUC', 'Max_Gr', 'CBF', 'CBV']] 
Y=df['Target'] 	
a=Data.shape[1]
X=Data[:-1]
X = np.asarray(X)
Y = np.asarray(Y).flatten()[:, None]
Y=Y[:-1]

ppv_list= []
npv_list = []
tp_list= []
fp_list = []
tn_list = []
fn_list = []
sensitivity_list= []
specificity_list= []

from sklearn.model_selection import KFold
from sklearn.utils import class_weight
kfold = KFold(n_splits=5, shuffle=True)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
acc_per_fold = []
loss_per_fold = []
X_train0, X_test, Y_train0, Y_test = train_test_split(X, Y, train_size=0.5, random_state=42,  stratify= Y, shuffle=True)  #do not go over 0.9
fold_no = 1   
for i in range(5):
  rs=i
  X_train, X_val, Y_train, Y_val = train_test_split(X_train0, Y_train0, train_size=0.7, random_state=rs, stratify= Y_train0, shuffle=True)
  sm = SMOTE(random_state=rs)
  X_train_sm, Y_train_sm = sm.fit_resample(X_train, Y_train)
  X_train=np.asarray(X_train_sm)
  X_test=np.asarray(X_test)
  Y_train=np.asarray(Y_train_sm)
  Y_test=np.asarray(Y_test)
  classes=np.unique(Y_train)
  neg=np.shape(Y_train[Y_train==0])[0]
  pos=np.shape(Y_train[Y_train==1])[0]
  weight_for_0 = (1 / neg) * ((neg+pos) / 2.0)
  weight_for_1 = (1 / pos) * ((neg+pos) / 2.0) 
  class_weights = {0: weight_for_0, 1: weight_for_1}
  
  model = Sequential()
  model.add(Dense(64, input_dim=a, activation='relu'))  
  model.add(BatchNormalization()) 
  model.add(Dropout(0.5))  
  model.add(Dense(32, activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Dropout(0.5))  
  model.add(Dense(16, activation='relu'))  
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))  
  opt = optimizers.Adam(learning_rate=0.001) #0.001
  model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
 
  weightfile = os.path.join('Weights',str(i))
  
  checkpoint = ModelCheckpoint(weightfile, save_weights_only=True, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
  
  es = EarlyStopping(monitor='val_loss', patience=50, mode='min', min_delta=1e-8)
  
  reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=20,
                              verbose=1, min_delta=1e-6, min_lr=0, cooldown=0)
  
  callbacks_list = [reduceLR, es, checkpoint]
  
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')
  
  history = model.fit(X_train, Y_train, batch_size=32,
               epochs=500, validation_data=(X_val, Y_val), class_weight= class_weights, callbacks=callbacks_list, verbose=1)
  scores = model.evaluate(X_test, Y_test, verbose=2)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])
  y_pred_keras = model.predict(X_test)
  pred=np.where(y_pred_keras > 0.51, 1,0)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')   
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  
  fpr, tpr, thresholds_keras = roc_curve(Y_test, pred)
  tprs.append(np.interp(mean_fpr, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  plt.plot(fpr, tpr, lw=1, alpha=0.3,
               label='ROC fold %d (AUC = %0.2f)' % (fold_no, roc_auc))
  
  plt.show()
  
  confusion_matrix = metrics.confusion_matrix(Y_test, pred) 
  tp= confusion_matrix[0,0]
  fp= confusion_matrix[0,1]
  fn= confusion_matrix[1,0]
  tn= confusion_matrix[1,1]
                                                        
  sensitivity= tp/(tp+fn)
  specificity= tn/(tn+fp)

  ppv= tp/(tp+fp)
  npv=tn/(tn+fn)
  
  tn_list.append(tn)
  fp_list.append(fp)

  fn_list.append(fn)
  tp_list.append(tp)

  sensitivity_list.append(sensitivity)
  specificity_list.append(specificity)

  ppv_list.append(ppv)
  npv_list.append(npv)

  fold_no = fold_no + 1
  tf.keras.backend.clear_session() 

print('------------------------------------------------------------------------')
print('Score per fold')

for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
  
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
          lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                  label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=8)
plt.ylabel('True Positive Rate',fontsize=8)
plt.title('Cross-Validation ROC ',fontsize=8)
plt.legend(loc="lower right", prop={'size': 8})
plt.show()

