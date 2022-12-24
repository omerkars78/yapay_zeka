import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

def eth_2019():
    eth_2019_csv = pd.read_table(r"eth-2019.csv",sep=(",")) # data.csv dosyasını oku
    eth_2019_list = eth_2019_csv.eth # eth sütununu listeye ata
    eth_2019_general_istatistic = eth_2019_csv.describe() # genel istatistikleri al
    # Veri çerçevesinden eğitim ve test verilerini ayırın
    x = eth_2019_csv[['Months']]  # Tahmin edilen değişkenler
    y = eth_2019_csv['eth']  # Tahmin edilen hedef değişken
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)   # 80% eğitim, 20% test

    # Lineer regresyon modelini oluşturun ve eğitin
    model = LinearRegression() # Modeli oluştur
    model.fit(X_train, y_train) # Modeli eğit

    # Modelin doğruluğunu test verileriyle ölçün
    accuracy = model.score(X_test, y_test) # Doğruluğu ölç
    print("Modelin doğruluğu:", accuracy) # Doğruluğu yazdır

    # Önümüzdeki 12 ay için tahminler yapın
    X_future = np.array(range(1, 13)).reshape(-1, 1) # 1-12 arasındaki sayıları diziye ata
    predictions = model.predict(X_future)
    print("Lineer Regresyona Göre Tahminler:", predictions)

def eth_2020():
    eth_2020_csv = pd.read_table(r"eth-2020.csv",sep=(",")) # data.csv dosyasını oku
    eth_2020_list = eth_2020_csv.eth # eth sütununu listeye ata
    eth_2020_general_istatistic = eth_2020_csv.describe() # genel istatistikleri al

    


def eth_2021():
    eth_2021_csv = pd.read_table(r"eth-2021.csv",sep=(",")) # data.csv dosyasını oku
    eth_2021_list = eth_2021_csv.eth # eth sütununu listeye ata
    eth_2021_general_istatistic = eth_2021_csv.describe() # genel istatistikleri al
    return eth_2021_list, eth_2021_general_istatistic

def eth_2022():
    eth_2022_csv = pd.read_table(r"eth-2022.csv",sep=(",")) # data.csv dosyasını oku
    eth_2022_list = eth_2022_csv.eth # eth sütununu listeye ata
    eth_2022_general_istatistic = eth_2022_csv.describe() # genel istatistikleri al
    return eth_2022_list, eth_2022_general_istatistic


eth_2019()