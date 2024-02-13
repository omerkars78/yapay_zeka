import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# Veri setini yükleme
data = pd.read_json('your_dataset.json')

# Tarih indeksini ayarlama ve veri setini eğitim/test olarak bölme
data.index = pd.to_datetime(data['Date'])
train = data['Value'][:'train_end_date']
test = data['Value']['test_start_date':]

# ARIMA modelini eğitme
model = ARIMA(train, order=(5,1,0))  # Bu parametreler veri setinize göre ayarlanmalıdır.
model_fit = model.fit(disp=0)

# Tahmin yapma
predictions = model_fit.forecast(steps=len(test))[0]

# Performansı değerlendirme
error = mean_squared_error(test, predictions)
print('Test MSE:', error)
