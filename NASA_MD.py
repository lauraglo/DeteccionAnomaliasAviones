import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.random import seed
from sktime.classification.compose import TimeSeriesForestClassifier

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

    def is_pos_def(A):
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False


    def cov_matrix(data, verbose=False):
        covariance_matrix = np.cov(data, rowvar=False)
        if is_pos_def(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if is_pos_def(inv_covariance_matrix):
                return covariance_matrix, inv_covariance_matrix


    def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
        inv_covariance_matrix = inv_cov_matrix
        vars_mean = mean_distr
        diff = data - vars_mean
        md = []
        for i in range(len(diff)):
            md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
        return md


    def MD_threshold(dist, extreme=False, verbose=False):
        k = 3. if extreme else 2.
        threshold = np.mean(dist) * k
        return threshold

    print(i)
    sample = pd.DataFrame(data[i], columns=["P30", "T30", "TGT", "FF"])
    sample.head()
    # Se usa el 70% de los datos para train y el 30% del final para test
    X_train = sample[0:round(len(sample) * 0.70)]
    X_test = sample [round(len(sample) * 0.70):]

    # Se comprimen los datos a componentes principales
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, svd_solver= 'full')
    X_train_PCA = pca.fit_transform(X_train)
    X_train_PCA = pd.DataFrame(X_train_PCA)
    X_train_PCA.index = X_train.index

    X_test_PCA = pca.transform(X_test)
    X_test_PCA = pd.DataFrame(X_test_PCA)
    X_test_PCA.index = X_test.index

    # Se utilizan los datos de PCA como valores de train/test
    data_train = np.array(X_train_PCA.values)
    data_test = np.array(X_test_PCA.values)

    # Calcular la matriz de covarianza
    cov_matrix, inv_cov_matrix  = cov_matrix(data_train)

    # Calcular la media de los datos de train para usar en la MD
    mean_distr = data_train.mean(axis=0)

    # Calcular distancias
    dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
    dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
    threshold = MD_threshold(dist_train, extreme = True)

    import seaborn as sns
    sns.set(color_codes=True)

    plt.figure()
    sns.distplot(np.square(dist_train),
                 bins = 10,
                 kde= False);
    plt.xlim([0.0,15])
    plt.figure()
    sns.distplot(dist_train,
                 bins = 10,
                 kde= True,
                color = 'green');
    #plt.xlim([0.0,5])
    #plt.xlabel('Mahalanobis dist')
    anomaly_train = pd.DataFrame()
    anomaly_train['Mob dist']= dist_train
    anomaly_train['Thresh'] = threshold

    # Si la MD es superior al umbral: anomalía
    anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
    anomaly_train.index = X_train_PCA.index
    anomaly = pd.DataFrame()
    anomaly['Mob dist']= dist_test
    anomaly['Thresh'] = threshold

    anomaly['Anomaly'] = anomaly['Mob dist'] > anomaly['Thresh']
    anomaly.index = X_test_PCA.index
    anomaly.head()

    anomaly_alldata = pd.concat([anomaly_train, anomaly])

    #anomaly_alldata.to_csv('Anomaly_distance.csv')
    anomaly_alldata.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])

    df2 = pd.DataFrame(anomaly_alldata['Mob dist'])
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
X_train, X_test, y_train, y_test = train_test_split(X2,y,random_state=15)

classifier = TimeSeriesForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))

print("-- Validacion CRUZADA --")
kf = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(classifier, X_train, y_train, cv=kf)

print("Metricas cross_validation", scores)

print("Media de cross_validation", scores.mean())

