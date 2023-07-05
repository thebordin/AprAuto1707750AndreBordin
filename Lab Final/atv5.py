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

df_ss = pd.read_csv(url_features_validation)

y_predicted = trickbag['MLPC'].predict(df_ss)
y_predicted = pd.DataFrame(y_predicted)
print(y_predicted)

