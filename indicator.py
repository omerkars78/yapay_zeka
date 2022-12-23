import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import mysql.connector

def a_coin_genel_istatistik():
        coinler_csv = pd.read_table(r"data.csv",sep=(";")) # data.csv dosyasını oku
        a_coin_liste=coinler_csv # a_coin_liste değişkenine ata
        a_coin_liste = coinler_csv.A_COİN # a_coin_liste değişkenine ata
        a_coin_genel_istatistik = a_coin_liste.describe() # a_coin_liste değişkeninin istatistiklerini a_coin_genel_istatistik değişkenine ata
        # np.array(a_coini_1_aylik_liste).max()
        # np.array(a_coini_1_aylik_liste).min()
        return a_coin_genel_istatistik # a_coin_genel_istatistik değişkenini döndür

def b_coin_genel_istatistik():
        coinler_csv = pd.read_table(r"data.csv",sep=(";")) # data.csv dosyasını oku
        b_coin_liste = coinler_csv.B_COİN # b_coin_liste değişkenine ata
        b_coin_genel_istatistik = b_coin_liste.describe() # b_coin_liste değişkeninin istatistiklerini b_coin_genel_istatistik değişkenine ata
#       np.array(b_coini_1_aylik_liste).max()
#       np.array(b_coini_1_aylik_liste).min()
        return b_coin_genel_istatistik # b_coin_genel_istatistik değişkenini döndür

def a_coin_5_gunluk_tahminler():
        coinler_csv = pd.read_table(r"data.csv",sep=(";")) # data.csv dosyasını oku
        a_coini_1_aylik_sozluk = dict(coinler_csv.A_COİN[:])  
        a_coini_1_aylik_liste = list(a_coini_1_aylik_sozluk.values()) # a_coini_1_aylik_sozluk değişkeninin değerlerini a_coini_1_aylik_liste değişkenine ata
        a_coini_dict = { # a_coini_1_aylik_sozluk değişkeninin anahtar ve değerlerini a_coini_dict değişkenine ata
            "gunler":list(a_coini_1_aylik_sozluk.keys()), 
            "degerler":list(a_coini_1_aylik_sozluk.values())     
            }
        a_coini_df = pd.DataFrame(data = a_coini_dict) # a_coini_dict değişkenini a_coini_df değişkenine ata
        gunler_a = a_coini_df[["gunler"]] # gunler_a değişkenine a_coini_df değişkeninin "gunler" sütununu ata
        degerler_a = a_coini_df[["degerler"]]   # degerler_a değişkenine a_coini_df değişkeninin "degerler" sütununu ata
        
        x_train_a,x_test_a,y_train_a,y_test_a=train_test_split(gunler_a,degerler_a,test_size=0.33,random_state=0) # x_train_a değişkenine x_test_a değişkenine y_train_a değişkenine y_test_a değişkenine ata
        
        poly_reg = PolynomialFeatures(degree=2)  # polynomal regresyonu 2. dereceden yap
        gunler_poly = poly_reg.fit_transform(gunler_a) # gunler_poly değişkenine gunler_a değişkenini polynomal regresyonu ile dönüştür
        
        lin_reg2=LinearRegression() # lin_reg2 değişkenine LinearRegression() fonksiyonunu ata lineer regresyonu yap
        lin_reg2.fit(gunler_poly,degerler_a)    # lin_reg2 değişkenine gunler_poly ve degerler_a değişkenlerini ata
        
        linear_reg = LinearRegression() # linear_reg değişkenine LinearRegression() fonksiyonunu ata lineer regresyonu yap
        linear_reg.fit(x_train_a,y_train_a) # linear_reg değişkenine x_train_a ve y_train_a değişkenlerini ata
        
        tahmin_1_a = linear_reg.predict([[32]]) # tahmin_1_a değişkenine linear_reg değişkeninin 32. günün tahmini değerini ata
        tahmin_2_a = linear_reg.predict([[33]]) # tahmin_2_a değişkenine linear_reg değişkeninin 33. günün tahmini değerini ata
        tahmin_3_a = linear_reg.predict([[34]]) # tahmin_3_a değişkenine linear_reg değişkeninin 34. günün tahmini değerini ata
        tahmin_4_a = linear_reg.predict([[35]]) # tahmin_4_a değişkenine linear_reg değişkeninin 35. günün tahmini değerini ata
        tahmin_5_a = linear_reg.predict([[36]]) # tahmin_5_a değişkenine linear_reg değişkeninin 36. günün tahmini değerini ata

        rf_reg = RandomForestRegressor(n_estimators=10,random_state=0) # rf_reg değişkenine RandomForestRegressor() fonksiyonunu ata karar ağacı regresyonu yap
        rf_reg.fit(gunler_a,degerler_a) 
       
        karar_agaci_a_1 = rf_reg.predict([[32]]) # karar_agaci_a_1 değişkenine rf_reg değişkeninin 32. günün tahmini değerini ata
        karar_agaci_a_2 = rf_reg.predict([[33]]) # karar_agaci_a_2 değişkenine rf_reg değişkeninin 33. günün tahmini değerini ata
        karar_agaci_a_3 = rf_reg.predict([[34]]) # karar_agaci_a_3 değişkenine rf_reg değişkeninin 34. günün tahmini değerini ata   
        karar_agaci_a_4 = rf_reg.predict([[35]]) # karar_agaci_a_4 değişkenine rf_reg değişkeninin 35. günün tahmini değerini ata
        karar_agaci_a_5 = rf_reg.predict([[36]]) # karar_agaci_a_5 değişkenine rf_reg değişkeninin 36. günün tahmini değerini ata
        
        polySonuc_1_a = lin_reg2.predict(poly_reg.fit_transform([[32]])) # polySonuc_1_a değişkenine lin_reg2 değişkeninin 32. günün tahmini değerini ata
        polySonuc_2_a = lin_reg2.predict(poly_reg.fit_transform([[33]])) # polySonuc_2_a değişkenine lin_reg2 değişkeninin 33. günün tahmini değerini ata   
        polySonuc_3_a = lin_reg2.predict(poly_reg.fit_transform([[34]])) # polySonuc_3_a değişkenine lin_reg2 değişkeninin 34. günün tahmini değerini ata
        polySonuc_4_a = lin_reg2.predict(poly_reg.fit_transform([[35]])) # polySonuc_4_a değişkenine lin_reg2 değişkeninin 35. günün tahmini değerini ata
        polySonuc_5_a = lin_reg2.predict(poly_reg.fit_transform([[36]])) # polySonuc_5_a değişkenine lin_reg2 değişkeninin 36. günün tahmini değerini ata

        print(f"Lineer Regresyona Göre A Coinin 5 Günlük Tahmin Verileri:{tahmin_1_a} {tahmin_2_a} {tahmin_3_a} {tahmin_4_a} {tahmin_5_a}")
        # Lineer Regresyona Göre 5 Günlük Tahmin Verileri
        
        print(f"Karar Ağacı Regresyona Göre A Coinin 5 Günlük Tahmin Verileri:{karar_agaci_a_1} {karar_agaci_a_2} {karar_agaci_a_3} {karar_agaci_a_4} {karar_agaci_a_5}")
        # Karar Ağacı Regresyona Göre 5 Günlük Tahmin Verileri

        print(f"Polinomal Regresyona Göre 5 A Coinin Günlük Tahmin Verileri:{polySonuc_1_a} {polySonuc_2_a} {polySonuc_3_a} {polySonuc_4_a} {polySonuc_5_a}")
        # Polinomal Regresyona Göre 5 Günlük Tahmin Verileri



def b_coin_5_gunluk_tahminler():
        coinler_csv = pd.read_table(r"data.csv",sep=(";")) # data.csv dosyasını oku
        b_coini_1_aylik_sozluk = dict(coinler_csv.B_COİN[:])  
        b_coini_1_aylik_liste = list(b_coini_1_aylik_sozluk.values()) # b_coini_1_aylik_sozluk değişkeninin değerlerini b_coini_1_aylik_liste değişkenine ata
        b_coini_dict = { # b_coini_1_aylik_sozluk değişkeninin anahtar ve değerlerini b_coini_dict değişkenine ata
            "gunler":list(b_coini_1_aylik_sozluk.keys()), 
            "degerler":list(b_coini_1_aylik_sozluk.values())     
            }
        b_coini_df = pd.DataFrame(data = b_coini_dict) # b_coini_dict değişkenini b_coini_df değişkenine ata
        gunler_b = b_coini_df[["gunler"]] # gunler_b değişkenine b_coini_df değişkeninin "gunler" sütununu ata
        degerler_b = b_coini_df[["degerler"]]   # degerler_b değişkenine b_coini_df değişkeninin "degerler" sütununu ata
        
        x_train_b,x_test_b,y_train_b,y_test_b=train_test_split(gunler_b,degerler_b,test_size=0.33,random_state=0) # x_train_b değişkenine x_test_b değişkenine y_train_b değişkenine y_test_b değişkenine ata
        
        poly_reg = PolynomialFeatures(degree=2)  # polynomal regresyonu 2. dereceden yap
        gunler_poly = poly_reg.fit_transform(gunler_b) # gunler_poly değişkenine gunler_b değişkenini polynomal regresyonu ile dönüştür
        
        lin_reg2=LinearRegression() # lin_reg2 değişkenine LinearRegression() fonksiyonunu ata lineer regresyonu yap
        lin_reg2.fit(gunler_poly,degerler_b)    # lin_reg2 değişkenine gunler_poly ve degerler_b değişkenlerini ata
        
        linear_reg = LinearRegression() # linear_reg değişkenine LinearRegression() fonksiyonunu ata lineer regresyonu yap
        linear_reg.fit(x_train_b,y_train_b) # linear_reg değişkenine x_train_b ve y_train_b değişkenlerini ata
        
        tahmin_1_b = linear_reg.predict([[32]]) # tahmin_1_b değişkenine linear_reg değişkeninin 32. günün tahmini değerini ata
        tahmin_2_b = linear_reg.predict([[33]]) # tahmin_2_b değişkenine linear_reg değişkeninin 33. günün tahmini değerini ata
        tahmin_3_b = linear_reg.predict([[34]]) # tahmin_3_b değişkenine linear_reg değişkeninin 34. günün tahmini değerini ata
        tahmin_4_b = linear_reg.predict([[35]]) # tahmin_4_b değişkenine linear_reg değişkeninin 35. günün tahmini değerini ata
        tahmin_5_b = linear_reg.predict([[36]]) # tahmin_5_b değişkenine linear_reg değişkeninin 36. günün tahmini değerini ata

        rf_reg = RandomForestRegressor(n_estimators=10,random_state=0) # rf_reg değişkenine RandomForestRegressor() fonksiyonunu ata karar ağacı regresyonu yap
        rf_reg.fit(gunler_b,degerler_b) 
       
        karar_agaci_b_1 = rf_reg.predict([[32]]) # karar_agaci_b_1 değişkenine rf_reg değişkeninin 32. günün tahmini değerini ata
        karar_agaci_b_2 = rf_reg.predict([[33]]) # karar_agaci_b_2 değişkenine rf_reg değişkeninin 33. günün tahmini değerini ata
        karar_agaci_b_3 = rf_reg.predict([[34]]) # karar_agaci_b_3 değişkenine rf_reg değişkeninin 34. günün tahmini değerini ata   
        karar_agaci_b_4 = rf_reg.predict([[35]]) # karar_agaci_b_4 değişkenine rf_reg değişkeninin 35. günün tahmini değerini ata
        karar_agaci_b_5 = rf_reg.predict([[36]]) # karar_agaci_b_5 değişkenine rf_reg değişkeninin 36. günün tahmini değerini ata
        
        polySonuc_1_b = lin_reg2.predict(poly_reg.fit_transform([[32]])) # polySonuc_1_b değişkenine lin_reg2 değişkeninin 32. günün tahmini değerini ata
        polySonuc_2_b = lin_reg2.predict(poly_reg.fit_transform([[33]])) # polySonuc_2_b değişkenine lin_reg2 değişkeninin 33. günün tahmini değerini ata   
        polySonuc_3_b = lin_reg2.predict(poly_reg.fit_transform([[34]])) # polySonuc_3_b değişkenine lin_reg2 değişkeninin 34. günün tahmini değerini ata
        polySonuc_4_b = lin_reg2.predict(poly_reg.fit_transform([[35]])) # polySonuc_4_b değişkenine lin_reg2 değişkeninin 35. günün tahmini değerini ata
        polySonuc_5_b = lin_reg2.predict(poly_reg.fit_transform([[36]])) # polySonuc_5_b değişkenine lin_reg2 değişkeninin 36. günün tahmini değerini ata

        print(f"Lineer Regresyona Göre B Coinin 5 Günlük Tahmin Verileri:{tahmin_1_b} {tahmin_2_b} {tahmin_3_b} {tahmin_4_b} {tahmin_5_b}")
        # Lineer Regresyona Göre 5 Günlük Tahmin Verileri

        print(f"Karar Ağacı Regresyona Göre B Coinin 5 Günlük Tahmin Verileri:{karar_agaci_b_1} {karar_agaci_b_2} {karar_agaci_b_3} {karar_agaci_b_4} {karar_agaci_b_5}")
        # Karar Ağacı Regresyona Göre 5 Günlük Tahmin Verileri
        
        print(f"Polinomal Regresyona Göre B Coinin 5 Günlük Tahmin Verileri:{polySonuc_1_b} {polySonuc_2_b}  {polySonuc_3_b}  {polySonuc_4_b}  {polySonuc_5_b}")
        # Polinomal Regresyona Göre 5 Günlük Tahmin Verileri


a_coin_5_gunluk_tahminler()