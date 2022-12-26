import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def year() -> None:
    # Veri kümenizi pandas veri çerçevesine yükleyin
    print()
    df = pd.read_csv("2018-2021.csv")
    df_2020 = pd.read_csv("2022.csv") 
    
    # Veri çerçevesinden eğitim ve test verilerini ayırın
    x = df[['days']]  # Tahmin edilen değişken
    y = df['eth']  # Tahmin edilen hedef değişken
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    # Lineer regresyon modelini oluşturun ve eğitin
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Modelin doğruluğunu test verileriyle ölçün
    accuracy = model.score(X_test, y_test)
    # print(f"Lineer regresyon modelinin doğruluğu: {accuracy:.3f}")

    # Önümüzdeki 12 ay için tahminler yapın
    X_future = np.array(range(1, 367)).reshape(-1, 1)
    predictions_2019 = model.predict(X_future)
    print("2019 Yılı Lineer regresyona göre tahminler:", predictions_2019)

    # 2. dereceden polinomal özellikler oluşturun
    poly = PolynomialFeatures(degree=2,include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # 2. dereceden polinomal regresyon modelini oluşturun ve eğitin
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)

    # Modelin doğruluğunu test verileriyle ölçün
    accuracy = model_poly.score(X_test_poly, y_test)
    # print(f"2. dereceden polinomal regresyon modelinin doğruluğu: {accuracy:.3f}")

    # Önümüzdeki 12 ay için tahminler yapın
    X_future_poly = poly.transform(X_future)
    predictions_poly = model_poly.predict(X_future_poly)
    
    # Karar Ağacı Modeli 
    model = RandomForestRegressor(n_estimators=100, random_state=0) # Modeli oluşturun
    model.fit(X_train, y_train)  # Modeli eğitin

    # Modelin doğruluğunu test verileriyle ölçün
    accuracy = model.score(X_test, y_test)
    # print(f"Karar ağacı modelinin doğruluğu: {accuracy:.3f}")

    # Önümüzdeki 12 ay için tahminler yapın
    X_future = np.array(range(1, 367)).reshape(-1, 1)
    predictions_2019_rf = model.predict(X_future)
    print("2019 Yılı Karar Ağacına göre tahminler:", predictions_2019_rf)
    
    print("2019 Yılı 2. dereceden polinomal regresyona göre tahminler:", predictions_poly)
    # print("*" * 50)
    print("2020 yılı gerçek verileri", df_2020.eth)
    # 2020 Tahmin Verileri ve Gerçek 2020 Verileri Karşılaştırması
    print("""
    
    2018-2022 VERİLERİNDEN TAHMİN EDİLEN 2022 VERİLERİ İLE GERÇEK 2020 VERİLERİNİN KARŞILAŞTIRILMASI
    
    """)
    total_difference = 0
    for i in range(len(predictions_poly)):
        if predictions_poly[i] != df_2020.eth[i]:
            difference = abs(predictions_poly[i] - df_2020.eth[i])
            percentage_difference = (difference / (predictions_poly[i] + df_2020.eth[i])) * 100
            total_difference += percentage_difference

            
            print("Index {} için % {} fark var.2020 Veri Karşılaştırması Polinomale göre".format(i, percentage_difference))
        else:
            print("Index {} için % 0 fark var.".format(i))
    average_difference = total_difference / len(predictions_poly)
    print("Ortalama % {} farklılık oranı var.".format(average_difference))
    print("""

    """)
    total_difference = 0
    for i in range(len(predictions_2019)):
        if predictions_2019[i] != df_2020.eth[i]:
            difference = abs(predictions_2019[i] - df_2020.eth[i])
            percentage_difference = (difference / (predictions_2019[i] + df_2020.eth[i])) * 100
            total_difference += percentage_difference
            
            print("Index {} için yüzde {} fark var.2020 Veri Karşılaştırması Lineere göre".format(i, percentage_difference))
        else:
            print("Index {} için yüzde 0 fark var.".format(i))
    average_difference = total_difference / len(predictions_poly)
    print("Ortalama % {} farklılık oranı var.".format(average_difference))
    print("""

    """)
    total_difference = 0
    for i in range(len(predictions_2019_rf)):
        if predictions_2019_rf[i] != df_2020.eth[i]:
            difference = abs(predictions_2019_rf[i] - df_2020.eth[i])
            percentage_difference = (difference / (predictions_2019_rf[i] + df_2020.eth[i])) * 100
            total_difference += percentage_difference
            
            print("Index {} için yüzde {} fark var.2020 Veri Karşılaştırması Karar Ağacına Göre".format(i, percentage_difference))
        else:
            print("Index {} için yüzde 0 fark var.".format(i))
    average_difference = total_difference / len(predictions_2019_rf)
    print("Ortalama % {} farklılık oranı var.".format(average_difference))