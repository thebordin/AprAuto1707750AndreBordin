import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

#### PARAMETROS ####
url_features_validation = 'data/heart_validation_3.csv'
rdm_state = 0
url_trickbag_std = 'data/predictorMLPC_heart_4.sav'
trickbag = joblib.load(open(url_trickbag_std, 'rb'))
#### FIM PARAMETROS ####

dataset = pd.read_csv(url_features_validation)
x_ss = dataset.iloc[:,:-1]
y_validation = dataset.iloc[:,-1]

y_predicted = trickbag['MLPC'].predict(x_ss)
y_predicted = pd.DataFrame(y_predicted)

cm = confusion_matrix(y_predicted,y_validation)

print('Traço = %.2f || Matriz confusão = %.2f || Score = %.2f || Precisao do MLPC = %.2f'% (cm.trace(), cm.sum(), trickbag['MLPC'].score(x_ss, y_validation), accuracy_score(y_validation, y_predicted)))
