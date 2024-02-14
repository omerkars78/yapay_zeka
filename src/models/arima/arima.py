import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Veri setini yükleme
data = pd.read_json('../../data/eth_new_json/eth.json')

# 'index' sütununu DataFrame'in indeksi olarak ayarlama
data.set_index('index', inplace=True)

# Veri setini eğitim seti olarak ayarlama
train = data['price']

# Gelecek 7 gün için tahminler yapma
future_predictions = []
for i in range(7):
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    prediction = model_fit.forecast()[0]  # Doğru şekilde tahmin al
    future_predictions.append(prediction)
    
    # Yeni tahmin edilen değeri eğitim verisine ekle
    new_index = train.index[-1] + pd.Timedelta(days=1)
    train = train.append(pd.Series(prediction, index=[new_index]))

print('Gelecek 7 günün tahminleri:', future_predictions)
