import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

#### PARAMETROS ####
url_features_validation = './data/features_validation.csv'
url_trickbag = './MLPC/MLPCpredictor.sav'
predictor = joblib.load(open(url_trickbag, 'rb'))
#### FIM PARAMETROS ####
print(f'===== Avaliação do modelo {url_trickbag} =====')
dataset = pd.read_csv(url_features_validation)
x_ss = dataset.iloc[:,:-1]
y_validation = dataset.iloc[:,-1]

y_predicted = predictor['MLPC'].predict(x_ss)
y_predicted = pd.DataFrame(y_predicted)

cm = confusion_matrix(y_predicted,y_validation)
print('Traço = %.2f || Matriz confusão = %.2f || Score = %.2f || Precisao do MLPC = %.2f'% (cm.trace(), cm.sum(), predictor['MLPC'].score(x_ss, y_validation), accuracy_score(y_validation, y_predicted)))
