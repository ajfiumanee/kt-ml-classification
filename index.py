# Imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Carrega arquivo

# Lê o arquivo utilizando as colunas informadas
dataset = pd.read_csv('dataset/kt-pentacan-dataset.csv', delimiter=',')

# dimensões do dataset
# print(dataset.shape)

# primeiras linhas do dataset
# print(dataset.head())

# Separação em conjuntos de treino e teste
array = dataset.values
columnsTemp = dataset.columns
X = array[:, 0:dataset.columns.size].astype(float)
Y = array[:, dataset.columns.size - 1]

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)
dec = clf.decision_function(X)
print(dec.shape[1])

clf = svm.SVC(decision_function_shape='ovr')
clf.fit(X, Y)
dec = clf.decision_function(X)
print(dec.shape[1])

stages = dataset.values[:, dataset.columns.size - 1:dataset.columns.size].astype(int)
stages = dataset
g1 = sns.displot(dataset, x="label")
g1.set_ylabels('Frequência')
g1.set_xlabels('Estágios')
plt.show()

test_size = 0.20
seed = 21
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Parâmetros
num_folds = 10
scoring = 'accuracy'

# Criação dos modelos
models = [
    ('LR', LogisticRegression(solver='newton-cg')),
    ('KNN', KNeighborsClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

# definindo uma semente global
np.random.seed(seed)

# Avaliação dos modelos
results = []
names = []
print('')

for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = " % s: % f (% f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Comparação dos modelos
fig = plt.figure()
fig.suptitle('Comparação dos Modelos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print('')

# Padronização do dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])))

results = []
names = []

for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train.ravel(), cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f" % (name, cv_results.mean())
    print(msg)

print('')

# Comparação dos modelos
fig = plt.figure()
fig.suptitle('Comparação dos Modelos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print('')

# Tuning do KNN
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
distancias = ["euclidean", "manhattan", "minkowski"]
param_grid = dict(n_neighbors=k, metric=distancias)

model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=1, verbose=1)
grid_result = grid.fit(rescaledX, Y_train)

print("Melhor: % f usando % s" % (grid_result.best_score_, grid_result.best_params_))

print('')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(" % f( % f): % r" % (mean, stdev, param))

# Tuning do SVM
c_values = [0.5, 1.0, 1.5, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)

model = SVC()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Melhor: %f com %s" % (grid_result.best_score_, grid_result.best_params_))

# print('')
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f): %r" % (mean, stdev, param))

print('')

# Preparação do modelo
model = LogisticRegression(solver='newton-cg')
model.fit(X_train, Y_train)

# Estimativa da acurácia no conjunto de teste
predictions = model.predict(X_test)
print("Accuracy score = ", accuracy_score(Y_test, predictions))

# print('')

# Matriz de confusão
cm = confusion_matrix(Y_test, predictions)
labels = ["1", "2", "3", "4"]
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot(values_format="d")
plt.suptitle('Matriz de confusão')
plt.show()
print(classification_report(Y_test, predictions, target_names=labels))
