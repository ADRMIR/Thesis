# Decision tree feature selection
##
##
#Python 3.7 by André Miranda @ andre.lima.miranda@tecnico.ulisboa.pt

# Part of the code retrieved from https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from Data_load import inputs, outputs

X_train=inputs.iloc[:600,:]
y_train=outputs['complicação pós-cirúrgica'].iloc[:600]

sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)

selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)



# feature_imp = pd.Series(sel.feature_importances_,index=inputs.columns).sort_values(ascending=False)
# feature_imp