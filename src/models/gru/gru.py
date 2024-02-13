from keras.layers import GRU

# GRU modeli (LSTM koduna çok benzer)
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(100,1)))
model.add(GRU(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Tahmin ve performans değerlendirme işlemleri LSTM ile aynıdır.
