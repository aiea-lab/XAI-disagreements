from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda
from keras.models import Model
from tensorflow.keras.layers import Embedding
from tqdm import tqdm
import shap
import lime
import lime.lime_tabular
import keras.utils
import csv
import numpy as np
import time

X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)

# normalize data (this is important for model convergence)
dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
for k,dtype in dtypes:
    if dtype == "float32":
        X[k] -= X[k].mean()
        X[k] /= X[k].std()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)

# build model
input_els = []
encoded_els = []
for k,dtype in dtypes:
    input_els.append(Input(shape=(1,)))
    if dtype == "int8":
        e = Flatten()(Embedding(X_train[k].max()+1, 1)(input_els[-1]))
    else:
        e = input_els[-1]
    encoded_els.append(e)
encoded_els = concatenate(encoded_els)
layer1 = Dropout(0.5)(Dense(100, activation="relu")(encoded_els))
out = Dense(1)(layer1)

# train model
regression = Model(inputs=input_els, outputs=[out])
regression.compile(optimizer="adam", loss='binary_crossentropy')
regression.fit(
    [X_train[k].values for k,t in dtypes],
    y_train,
    epochs=50,
    batch_size=512,
    shuffle=True,
    validation_data=([X_valid[k].values for k,t in dtypes], y_valid)
)

def f(X):
    return regression.predict([X[:,i] for i in range(X.shape[1])]).flatten()

# Explainer Training Data
x_train_shuffle = np.copy(X_train.values)
np.random.shuffle(x_train_shuffle)
percent = [0.01, 0.05, 0.1, 0.25]
for current_percent in percent:
    starting_time = time.time()
    file_name = 'explanation_model_'+str(current_percent)+'.csv'
    exp_train = x_train_shuffle[:int(current_percent*len(X_train.values))]

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(exp_train, feature_names=list(X_display.columns), class_names=['under 50k', 'over 50k'], mode='regression')
    csv_results = {}
    compiled_results = []
    csv_results['model'] = 'lime'
    field_names = ['model', 'index', 'Age', 'Workclass', 'Education-Num', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week', 'Country']
    for i in range(len(X_valid.values)):
        
        csv_results['index'] = i
        if i%(len(X_valid.values)//100) == 0:
            print(csv_results['model'], csv_results['index'], current_percent, i/len(X_valid.values), starting_time-time.time())
        lime_exp = lime_explainer.explain_instance(X_valid.values[i,:], f, num_features=12)
        for count, j in enumerate(lime_exp.as_list()):
            csv_results[field_names[count+2]] = j[1]
        compiled_results.append(csv_results.copy())

    with open(file_name, 'w', newline='') as csv_file:
        dict_obj = csv.DictWriter(csv_file, fieldnames=field_names)
        dict_obj.writeheader
        for csv_row in compiled_results:
            dict_obj.writerow(csv_row)
        csv_file.close()

    explainer = shap.KernelExplainer(f, exp_train)
    csv_results['model'] = 'shap'
    shap_exp = explainer.shap_values(X_valid.values, nsamples=500)
    for i, shap_result in enumerate(shap_exp):
        csv_results['index'] = i
        if i%(len(X_valid.values)//100) == 0:
            print(csv_results['model'], csv_results['index'], current_percent, i/len(X_valid.values), starting_time-time.time())
        for count, j in enumerate(shap_result):
            csv_results[field_names[count+2]] = j
        compiled_results.append(csv_results.copy())

    with open(file_name, 'a', newline='') as csv_file:
        dict_obj = csv.DictWriter(csv_file, fieldnames=field_names)
        for csv_row in compiled_results:
            dict_obj.writerow(csv_row)
        csv_file.close()