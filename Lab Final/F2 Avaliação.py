import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import sys

#### PARAMETROS ####
url_features_validation = './data/features_validation.csv'
url_trickbag = './MLPC/MLPCpredictor.sav'
predictor = joblib.load(open(url_trickbag, 'rb'))
#### FIM PARAMETROS ####

dataset = pd.read_csv(url_features_validation)
x_ss = dataset.iloc[:,:-1]
y_validation = dataset.iloc[:,-1]

y_predicted = predictor['MLPC'].predict(x_ss)
y_predicted = pd.DataFrame(y_predicted)

cm = confusion_matrix(y_predicted,y_validation)
print(f'Validation: \n{y_validation.T}####### \n Predicted: \n{y_predicted.T}')
print('Traço = %.2f || Matriz confusão = %.2f || Score = %.2f || Precisao do MLPC = %.2f'% (cm.trace(), cm.sum(), predictor['MLPC'].score(x_ss, y_validation), accuracy_score(y_validation, y_predicted)))
