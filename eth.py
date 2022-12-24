import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def eth_2019() -> None:
    # Veri kümenizi pandas veri çerçevesine yükleyin
    print("*" * 50)
    df = pd.read_csv("eth-2019.csv")

    # Veri çerçevesinden eğitim ve test verilerini ayırın
    x = df[['Months']]  # Tahmin edilen değişken
    y = df['eth']  # Tahmin edilen hedef değişken
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    # Lineer regresyon modelini oluşturun ve eğitin
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Modelin doğruluğunu test verileriyle ölçün
    accuracy = model.score(X_test, y_test)
    # print(f"Lineer regresyon modelinin doğruluğu: {accuracy:.3f}")

    # Önümüzdeki 12 ay için tahminler yapın
    X_future = np.array(range(1, 13)).reshape(-1, 1)
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
    print("2019 Yılı 2. dereceden polinomal regresyona göre tahminler:", predictions_poly)
    print("*" * 50)

def eth_2020() -> None:
    print("*" * 50)
    # Veri kümenizi pandas veri çerçevesine yükleyin
    df = pd.read_csv("eth-2020.csv")

    # Veri çerçevesinden eğitim ve test verilerini ayırın
    x = df[['Months']]  # Tahmin edilen değişken
    y = df['eth']  # Tahmin edilen hedef değişken
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    # Lineer regresyon modelini oluşturun ve eğitin
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Modelin doğruluğunu test verileriyle ölçün
    accuracy = model.score(X_test, y_test)
    # print(f"Lineer regresyon modelinin doğruluğu: {accuracy:.3f}")

    # Önümüzdeki 12 ay için tahminler yapın
    X_future = np.array(range(1, 13)).reshape(-1, 1)
    predictions_2020 = model.predict(X_future)
    print("2020 Yılı Lineer regresyona göre tahminler:", predictions_2020)

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
    predictions_poly_2020 = model_poly.predict(X_future_poly)
    print("2020 Yılı 2. dereceden polinomal regresyona göre tahminler:", predictions_poly_2020)
    print("*" * 50)

def eth_2021() -> None:
    print("*" * 50)
    # Veri kümenizi pandas veri çerçevesine yükleyin
    df = pd.read_csv("eth-2021.csv")

    # Veri çerçevesinden eğitim ve test verilerini ayırın
    x = df[['Months']]  # Tahmin edilen değişken
    y = df['eth']  # Tahmin edilen hedef değişken
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    # Lineer regresyon modelini oluşturun ve eğitin
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Modelin doğruluğunu test verileriyle ölçün
    accuracy = model.score(X_test, y_test)
    # print(f"Lineer regresyon modelinin doğruluğu: {accuracy:.3f}")

    # Önümüzdeki 12 ay için tahminler yapın
    X_future = np.array(range(1, 13)).reshape(-1, 1)
    predictions_2021 = model.predict(X_future)
    print("2021 Yılı Lineer regresyona göre tahminler:", predictions_2021)

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
    predictions_poly_2021 = model_poly.predict(X_future_poly)
    print("2021 Yılı 2. dereceden polinomal regresyona göre tahminler:", predictions_poly_2021)
    print("*" * 50)

def eth_2022() -> None:
    print("*" * 50)
    # Veri kümenizi pandas veri çerçevesine yükleyin
    df = pd.read_csv("eth-2022.csv")

    # Veri çerçevesinden eğitim ve test verilerini ayırın
    x = df[['Months']]  # Tahmin edilen değişken
    y = df['eth']  # Tahmin edilen hedef değişken
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    # Lineer regresyon modelini oluşturun ve eğitin
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Modelin doğruluğunu test verileriyle ölçün
    accuracy = model.score(X_test, y_test)
    # print(f"Lineer regresyon modelinin doğruluğu: {accuracy:.3f}")

    # Önümüzdeki 12 ay için tahminler yapın
    X_future = np.array(range(1, 13)).reshape(-1, 1)
    predictions_2022 = model.predict(X_future)
    print("2022 Yılı Lineer regresyona göre tahminler:", predictions_2022)

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
    predictions_poly_2022 = model_poly.predict(X_future_poly)
    print("2022 Yılı 2. dereceden polinomal regresyona göre tahminler:", predictions_poly_2022)   
    print("*" * 50)




eth_2019()
eth_2020()
eth_2021()
eth_2022()