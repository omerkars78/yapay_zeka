import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler

# Veri setini yükleme
data = pd.read_json('../../data/eth_new_json/eth.json')

# 'price' sütununu kullanarak veriyi normalizasyon
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['price'].values.reshape(-1,1))

# Eğitim verilerini ayarlama
time_step = 100
train_data = scaled_data[:-7]

# Veri setini GRU modeli için uygun hale getirme
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data, time_step)

# GRU modelini oluşturma ve eğitme
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(GRU(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Gelecek 7 gün için tahminler yapma
future_predictions = []
new_input = train_data[-time_step:]

for _ in range(7):
    # Yeni veri seti ile tahmin yapma
    new_input = new_input.reshape((1, time_step, 1))
    prediction = model.predict(new_input)
    
    # Tahmin edilen değeri gelecek tahminler için veri setine ekleme
    new_input = np.append(new_input, prediction)[-time_step:]
    
    # Tahmini gerçek değerlere dönüştürme ve kaydetme
    inverse_prediction = scaler.inverse_transform(prediction)
    future_predictions.append(inverse_prediction[0, 0])

print('Gelecek 7 günün tahminleri:', future_predictions)
