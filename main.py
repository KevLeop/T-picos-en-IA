from scipy.stats import randint
from sklearn import metrics # para revisar el error y precision del modelo
from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler # para normalizacion
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline # pipeline: aplica una transformacion a los datos
from keras.layers.core import Dense, Activation, Dropout
import numpy as np
import sys 
import matplotlib.pyplot as plt #
import pandas as pd
import itertools
import keras
from keras.models import Sequential  #pila lineal de capas
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD 
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D

def global_active_power_grafico(dataset):
	dataset.energia_activa.resample('D').sum().plot(title='Energia activa por dia') 
	plt.tight_layout()
	plt.show()

dataset = pd.read_csv('data.txt', sep=';', parse_dates={'dt' : ['Fecha', 'Hora']}, infer_datetime_format=True, 
                  low_memory=False, na_values=['nan','?'], index_col='dt')

global_active_power_grafico(dataset)

values = dataset.resample('D').mean().values


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	nuevo_dataset = pd.DataFrame(data)
	cols, names = list(), list()
	# secuencia de entrada (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(nuevo_dataset.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# secuencia de pronóstico (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(nuevo_dataset.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# juntando todos los datos
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# elimina filas con valores NaN
	if dropnan:
		agg.dropna(inplace=True)
	return agg



dataset_resample = dataset.resample('h').mean() 
dataset_resample.shape
values = dataset_resample.values 

# normalizar caracteristicas
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

# marco como aprendizaje supervisado
reframed = series_to_supervised(scaled, 1, 1)

# eliminamos lo que no queremos predecir
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(reframed.head())


# particion en conjuntos de entrenmiento y prueba
values = reframed.values

n_train_time = 365 * 24 * 4#24
train = values[:n_train_time, :]
test = values[n_train_time:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# ajustamos la red
modelo = model.fit(train_X, train_y, epochs=20, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# grafico entrenamiento y prueba
plt.plot(modelo.history['loss'])
plt.plot(modelo.history['val_loss'])
plt.title('Error')
plt.ylabel('error')
plt.xlabel('iteracion')
plt.legend(['entrenamiento', 'prueba'], loc='upper right')
plt.show()

# prediccion
yhat = model.predict(test_X, verbose=0)
test_X = test_X.reshape((test_X.shape[0], 7))

inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

rmse = np.sqrt(mean_squared_error(test_y, yhat))
print('RMSE: %.3f' % rmse)

aa=[x for x in range(400)]
bb=[x for x in range(300)]
plt.plot(bb, inv_y[:300], marker='.', label="actual")
plt.plot(aa, inv_yhat[:400], 'r', label="prediccion")
plt.legend(fontsize=15)
plt.xlabel('Tiempo', size=15)
plt.ylabel('Energia Activa', size=15)
plt.show()


