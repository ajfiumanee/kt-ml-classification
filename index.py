# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Carrega arquivo

# Lê o arquivo utilizando as colunas informadas
dataset = pd.read_csv('dataset/kt-pentacan-dataset.csv', delimiter=',')

dataset['LogMAR UDVA'] = dataset['LogMAR UDVA'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['LogMAR CDVA'] = dataset['LogMAR CDVA'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['Sphere'] = dataset['Sphere'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['Cylinder'] = dataset['Cylinder'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['Flat Keratometry'] = dataset['Flat Keratometry'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['Steep Keratometry'] = dataset['Steep Keratometry'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['Steep Axis'] = dataset['Steep Axis'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['Mean Topograpjy K'] = dataset['Mean Topograpjy K'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['Topography Cylinder'] = dataset['Topography Cylinder'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['Location X Axis'] = dataset['Location X Axis'].apply(lambda x: str(x).replace(',', '.')).astype(float)
dataset['Location Y Axis'] = dataset['Location Y Axis'].apply(lambda x: str(x).replace(',', '.')).astype(float)

# dimensões do dataset
# print(dataset.shape)

# primeiras linhas do dataset
# print(dataset.head())

# Estágios
stages = dataset.values[:, dataset.columns.size - 1:dataset.columns.size].astype(int)
# print(str(stages))

# g1 = sns.displot(dataset, x="label")
# g1.set_ylabels('Frequência')
# g1.set_xlabels('Estágios')
# plt.show()

###########

# Separação em conjuntos de treino e teste
array = dataset.values

X = dataset.drop(['General Health', 'Atopy',
                  'Hypertension', 'Hayfever', 'Known Eye History',
                  'Family History KC', 'Primary Optical Aid', 'Eye', 'Sphere', 'Cylinder', 'Axis',
                  'Flat Keratometry', 'Steep Keratometry', 'Steep Axis',
                  'Topography Cylinder', 'Central Pachy',
                  'Thinnest pachy', 'Location X Axis', 'Location Y Axis'], axis=1)
Y = dataset["label"]

# Parâmetros
test_size = 0.20
seed = 21
seed_global = 7
num_folds = 12
num_jobs = 2
scoring = 'accuracy'
solver = 'newton-cg'
max_interactions = 10000

###########

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, shuffle='true')

# Criação dos modelos
models = [
    ('LR', LogisticRegression(solver=solver)),
    ('KNN', KNeighborsClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

###########

# definindo uma semente global
np.random.seed(seed_global)

# Avaliação dos modelos
results = []
names = []
print('')

print('Models:')
for name, model in models:
    kFoldModel = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kFoldModel, scoring=scoring, n_jobs=num_jobs)
    results.append(cv_results)
    names.append(name)
    msg = " % s: % f (% f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print('')

# Comparação dos modelos
fig = plt.figure()
# fig.suptitle('Comparação dos Modelos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print('')

###########

pipelinesResults = []
pipelinesNames = []

# Padronização do dataset

np.random.seed(seed_global)

pipelines = [('Scaled LR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])),
             ('Scaled KNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])),
             ('Scaled NB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])),
             ('Scaled SVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())]))]

print('Pipelines:')
for name, model in pipelines:
    kFoldItem = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train.ravel(), cv=kFoldItem, scoring=scoring)
    pipelinesResults.append(cv_results)
    pipelinesNames.append(name)
    msg = "%s: %f (% f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print('')

# Comparação dos Pipeline
fig = plt.figure()
# fig.suptitle('Comparação do Pipeline')
ax = fig.add_subplot(111)
plt.boxplot(pipelinesResults)
ax.set_xticklabels(pipelinesNames)
plt.show()

print('')

###########

# Tuning do KNN

np.random.seed(seed_global)

# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)

k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
metrics = ["euclidean", "manhattan", "minkowski"]
param_grid = dict(n_neighbors=k, metric=metrics)

model = KNeighborsClassifier()
kFold = KFold(n_splits=num_folds)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kFold, n_jobs=num_jobs, verbose=1)
grid_result = grid.fit(X_train, Y_train)

print('KNN')
print("Melhor: % f usando % s" % (grid_result.best_score_, grid_result.best_params_))

print('')
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print(" % f( % f): % r" % (mean, stdev, param))
#
# print('')

###########

np.random.seed(seed_global)

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

# Tuning do SVM
c_values = [0.1, 0.5, 1.0, 1.5, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)

model = SVC()
kFold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kFold)
grid_result = grid.fit(rescaledX, Y_train)
print('SVM')
print("Melhor: %f com %s" % (grid_result.best_score_, grid_result.best_params_))

print('')
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f): %r" % (mean, stdev, param))
#
# print('')

###########

np.random.seed(seed_global)

# Preparação do modelo

scaler = StandardScaler().fit(X)
X_standardized = scaler.transform(X)

X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(X_standardized, Y, test_size=test_size, random_state=seed,
                                                            shuffle='true', stratify=Y)


# model = LogisticRegression(solver=solver, random_state=seed)
model = SVC(C=0.5, random_state=seed, kernel='linear')

model.fit(X_train_S, y_train_S)

# Estimativa da acurácia no conjunto de teste
predictions = model.predict(X_test_S)
print("Accuracy score = ", accuracy_score(y_test_S, predictions))

print('')

# Matriz de confusão
cm = confusion_matrix(y_test_S, predictions)
labels = ["1", "2", "3", "4"]
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot(values_format="d")
plt.suptitle('Matriz de confusão - SVM')
plt.show()

print(classification_report(y_test_S, predictions, target_names=labels, zero_division=0))
