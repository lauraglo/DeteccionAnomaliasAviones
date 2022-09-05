import numpy as np
import pandas as pd
import sns as sns
from keras import regularizers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.random import seed
from sktime.classification.compose import TimeSeriesForestClassifier
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from keras.layers import Input, Dropout
from keras.layers.core import Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json
train_df = pd.read_csv("./CMAPSSData/test_FD001.txt",sep=" ",header=None)

train_df.drop(columns=[26,27],inplace=True)

columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]

train_df.columns = columns

sub_train_df = train_df[["unit_number","P30", "T30", "epr","T50"]]
sub_train_df = sub_train_df.rename(columns={'epr': 'EPRA', 'T50': 'TGT'})
sub_train_df["FF"] = train_df["phi"] * train_df["Ps30"]
sub_train_df.head()

np.unique(sub_train_df["EPRA"]) # Datos de EPRA ctes

# Como los valores de EPRA son todos iguales no aportan información, se prescinde de ellos

sub_train_df.drop(['EPRA'],axis=1)

scaler = MinMaxScaler()
scaler.fit(sub_train_df[["P30", "T30", "TGT","FF"]])
minmax_data = scaler.transform(sub_train_df[["P30", "T30", "TGT","FF"]])

n_samples = len(np.unique(sub_train_df["unit_number"]))

norm_data =[]

for sample in range(1, n_samples+1):
  width = sub_train_df.index[sub_train_df["unit_number"]==sample].tolist()
  norm_data.append(minmax_data[width[0]:width[-1]+1])

# --- SUAVIZADO ---
# Suavizado utilizando el 3% de datos como ventana
def smooth(data):
    import numpy as np
    from scipy.signal import savgol_filter
    window_length = round(0.03 * len(data))
    if window_length % 2 == 0:
        window_length = window_length + 1
    smooth_pre = savgol_filter(data, window_length, 2)
    return savgol_filter(smooth_pre, window_length, 2)

# Aquí los índices son de los aviones, no del array, así que van 1 por delante (no empiezan en 0 como en data)
plt.plot(sub_train_df[sub_train_df["unit_number"]==100]["P30"])

data = norm_data.copy()



plt.plot(data[99][:,0])

data = np.array(data)

np.save("./CMAPSSData/test_FD001.npy", data)


data = np.load('./CMAPSSData/test_FD001.npy', allow_pickle=True) #data[0] -> primer motor

y = []
RUL = pd.read_csv(('./CMAPSSData/RUL_FD001.txt'), sep='\s+', header=None, names=['RemainingUsefulLife'])
vecrul = RUL['RemainingUsefulLife']


for i in vecrul:
    if i<40:
        y.append(1)
    else:
        y.append(0)


X = np.zeros((100,50))

i = 0
#recorremos todos los motores
while(i < len(data)):
    print(i)
    sample = pd.DataFrame(data[i], columns=["P30", "T30", "TGT", "FF"])
    sample.head()
    # Se usa el 70% de los datos para train y el 30% del final para test
    X_train = sample[0:round(len(sample) * 0.70)]
    X_test = sample[round(len(sample) * 0.70):]

    # Se comprimen los datos a componentes principales
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, svd_solver='full')
    X_train_PCA = pca.fit_transform(X_train)
    X_train_PCA = pd.DataFrame(X_train_PCA)
    X_train_PCA.index = X_train.index

    X_test_PCA = pca.transform(X_test)
    X_test_PCA = pd.DataFrame(X_test_PCA)
    X_test_PCA.index = X_test.index

    # Se utilizan los datos de PCA como valores de train/test
    data_train = np.array(X_train_PCA.values)
    data_test = np.array(X_test_PCA.values)

    seed(10)
    # set_random_seed(10)
    act_func = 'elu'

    # Input layer:
    model = Sequential()
    # 1º capa oculta
    # First hidden layer, connected to input vector X.
    model.add(Dense(10, activation=act_func,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0.0),
                    input_shape=(X_train.shape[1],)
                    )
              )

    model.add(Dense(2, activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(10, activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(X_train.shape[1],
                    kernel_initializer='glorot_uniform'))

    model.compile(loss='mse', optimizer='adam')

    # Train model for 100 epochs, batch size of 10:
    NUM_EPOCHS = 100
    BATCH_SIZE = 10

    # Ajustamos el modelo para usar el 5% del conjunto de entrenamiento para la validacion (0.05)
    # Fitting the model:
    # To keep track of the accuracy during training, we use 5% of the training data for validation after each epoch (validation_split = 0.05)
    history = model.fit(np.array(X_train), np.array(X_train),
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_split=0.05,
                        verbose=1)

    # Visualize training/validation loss:
    plt.plot(history.history['loss'],
             'b',
             label='Training loss')
    plt.plot(history.history['val_loss'],
             'r',
             label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    plt.ylim([0, .1])
    # plt.show()

    # Si dibujamos la perdida podemos identificar el valor límite
    # Distribution of loss function in the training set:
    # By plotting the distribution of the calculated loss in the training set, one can use this to identify a
    # suitable threshold value for identifying an anomaly. In doing this, one can make sure that this threshold
    # is set above the “noise level”, and that any flagged anomalies should be statistically significant above the noise background.

    X_pred = model.predict(np.array(X_train))
    X_pred = pd.DataFrame(X_pred,
                          columns=X_train.columns)
    X_pred.index = X_train.index

    scored = pd.DataFrame(index=X_train.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred - X_train), axis=1)
    # plt.figure()


    # From the above loss distribution, let us try a threshold of 0.3 for flagging an anomaly. We can then calculate
    # the loss in the test set, to check when the output crosses the anomaly threshold.

    X_pred = model.predict(np.array(X_test))
    X_pred = pd.DataFrame(X_pred,
                          columns=X_test.columns)
    X_pred.index = X_test.index

    scored = pd.DataFrame(index=X_test.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred - X_test), axis=1)
    scored['Threshold'] = 0.3
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    scored.head()

    # We then calculate the same metrics also for the training set, and merge all data in a single dataframe:
    X_pred_train = model.predict(np.array(X_train))
    X_pred_train = pd.DataFrame(X_pred_train,
                                columns=X_train.columns)
    X_pred_train.index = X_train.index

    scored_train = pd.DataFrame(index=X_train.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train - X_train), axis=1)
    scored_train['Threshold'] = 0.3
    scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
    scored = pd.concat([scored_train, scored])

    df2 = pd.DataFrame(scored['Loss_mae'])


    vec = df2.to_numpy()
    vec = vec.flatten()

    # X[cont] = vec -> ValueError: could not broadcast input array from shape (4425) into shape (5000)
    # antes de X[cont] = vec, tenemos que ajustar todos los vectores para que su longitud sea igual

    indint = np.round(np.linspace(0, len(vec) - 1, 50, True))  # indices del vector reducido (1000 valores)

    xp = np.arange(0, len(vec), 1)

    vec2 = np.interp(indint, xp, vec)
    X[i] = vec2

    i = i + 1


index = np.arange(0,100)
X2 = pd.DataFrame([[i] for i in X],index = index)

from tslearn.utils import to_time_series_dataset
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X2,y,random_state=42)

classifier = TimeSeriesForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print("-- Validacion CRUZADA --")

kf = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(classifier, X_train, y_train, cv=kf, scoring="accuracy")

print("Metricas cross_validation", scores)

print("Media de cross_validation", scores.mean())

