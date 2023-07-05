import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import sys

#### PARAMETROS ####
url_features_validation = sys.argv[1]
url_trickbag_std = sys.argv[2]
rdm_state = 42
trickbag = joblib.load(open(url_trickbag_std, 'rb'))
#### FIM PARAMETROS ####

dataset = pd.read_csv(url_features_validation)
x_ss = dataset.iloc[:,:-1]
y_validation = dataset.iloc[:,-1]

y_predicted = trickbag['MLPC'].predict(x_ss)
y_predicted = pd.DataFrame(y_predicted)

cm = confusion_matrix(y_predicted,y_validation)

print('Traço = %.2f || Matriz confusão = %.2f || Score = %.2f || Precisao do MLPC = %.2f'% (cm.trace(), cm.sum(), trickbag['MLPC'].score(x_ss, y_validation), accuracy_score(y_validation, y_predicted)))
