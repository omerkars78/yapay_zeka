import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
# import matplotlib.pyplot as plt
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# import mysql.connector
# class coinler(QDialog):
#     def __init__(self,parent=None):
#         super(coinler,self).__init__(parent)
#         grid = QGridLayout()        
#         grid=QGridLayout()
    
#         grid.addWidget(QLabel("<font color = 'red' font size = '+5'>""A_COİN ve B_COİN 1 AYLIK VERİLER VE İSTATİSTİKLER</font>"),0,0,1,2)
#         grid.addWidget(QLabel("<font color = 'blue' font size = '+3'> A_COİN ve B_COİN 1 AYLIK VERİLER </font>"),1,0)
#         grid.addWidget(QLabel("<font color = 'green' font size = '+3'> A_COİN ve B_COİN 1 AYLIK GRAFİK </font>"),1,2)
#         grid.addWidget(QLabel("<font color = 'purple' font size = '+3'> A_COİN ve B_COİN GENEL İSTATİSTİKİ VERİLER </font>"),4,0)
#         grid.addWidget(QLabel("<font color = 'orange' font size = '+3'> A_COİN ve B_COİN 5 GÜNLÜK TAHMİN DEĞERLERİ </font>"),4,2)
       
#         self.a_b_coinleri_listele = QTextEdit()
#         grid.addWidget(self.a_b_coinleri_listele,3,0)
        
#         self.a_coin_genel_istatistik = QTextEdit()
#         grid.addWidget(self.a_coin_genel_istatistik,6,0)
      
#         self.b_coin_genel_istatistik = QTextEdit()
#         grid.addWidget(self.b_coin_genel_istatistik,6,1)
        
#         self.grafik_a = QLabel()
#         self.grafik_b = QLabel()
#         grid.addWidget(self.grafik_a,3,2)
#         grid.addWidget(self.grafik_b,3,3)
       
#         grid.addWidget(QLabel("<font color = 'red' font size = '+1'> A_COİN Tahminleri </font>"),6,2,1,2)
          
#         self.a_coin_tahmin_1 = QLabel()
#         self.a_coin_tahmin_2 = QLabel()
#         self.a_coin_tahmin_3 = QLabel()
#         self.a_coin_tahmin_4 = QLabel()
#         self.a_coin_tahmin_5 = QLabel()
#         grid.addWidget(self.a_coin_tahmin_1,7,2)
#         grid.addWidget(self.a_coin_tahmin_2,8,2)
#         grid.addWidget(self.a_coin_tahmin_3,9,2)
#         grid.addWidget(self.a_coin_tahmin_4,10,2)
#         grid.addWidget(self.a_coin_tahmin_5,11,2)
        
#         grid.addWidget(QLabel("<font color = 'red' font size = '+1'> B_COİN Tahminleri </font>"),6,3)
#         self.b_coin_tahmin_1 = QLabel()
#         self.b_coin_tahmin_2 = QLabel()
#         self.b_coin_tahmin_3 = QLabel()
#         self.b_coin_tahmin_4 = QLabel()
#         self.b_coin_tahmin_5 = QLabel()
#         grid.addWidget(self.b_coin_tahmin_1,7,3)
#         grid.addWidget(self.b_coin_tahmin_2,8,3)
#         grid.addWidget(self.b_coin_tahmin_3,9,3)
#         grid.addWidget(self.b_coin_tahmin_4,10,3)
#         grid.addWidget(self.b_coin_tahmin_5,11,3)
        
#         a_b_coin_liste_buton = QPushButton("A_COİN ve B_COİN 1 Aylık Listele")
#         grid.addWidget(a_b_coin_liste_buton,2,0)
#         a_b_coin_liste_buton.clicked.connect(self.a_b_coin_liste_buton) 
        
#         a_coin_grafik_buton = QPushButton("A_COİN Grafik")
#         grid.addWidget(a_coin_grafik_buton,2,2)
#         a_coin_grafik_buton.clicked.connect(self.a_coin_grafik_buton)
        
#         b_coin_grafik_buton = QPushButton("B_COİN Grafik")
#         grid.addWidget(b_coin_grafik_buton,2,3)
#         b_coin_grafik_buton.clicked.connect(self.b_coin_grafik_buton)
    
#         a_coin_genel_istatistik_buton = QPushButton("A_COİN Genel Veriler")
#         grid.addWidget(a_coin_genel_istatistik_buton,5,0)
#         a_coin_genel_istatistik_buton.clicked.connect(self.a_coin_genel_istatistik_buton)
 
#         b_coin_genel_istatistik_buton = QPushButton("B_COİN Genel Veriler")
#         grid.addWidget(b_coin_genel_istatistik_buton,5,1)
#         b_coin_genel_istatistik_buton.clicked.connect(self.b_coin_genel_istatistik_buton)
 
#         a_coin_5_gunluk_tahmin_buton = QPushButton("A_COİN 5 Günlük Tahmin")
#         grid.addWidget(a_coin_5_gunluk_tahmin_buton,5,2)
#         a_coin_5_gunluk_tahmin_buton.clicked.connect(self.a_coin_5_gunluk_tahmin_buton)
      
#         b_coin_5_gunluk_tahmin_buton = QPushButton("B_COİN 5 Günlük Tahmin")
#         grid.addWidget(b_coin_5_gunluk_tahmin_buton,5,3)
#         b_coin_5_gunluk_tahmin_buton.clicked.connect(self.b_coin_5_gunluk_tahmin_buton)
        
#         tahminleri_kaydet_buton = QPushButton("Tahmin Verilerini Kaydet")
#         grid.addWidget(tahminleri_kaydet_buton,12,3)
#         tahminleri_kaydet_buton.clicked.connect(self.tahminleri_kaydet_buton)
#         self.setWindowTitle("A_COİN VE B_COİN 1 AYLIK")    
#         self.setLayout(grid)
    
#     def a_b_coin_liste_buton(self):
#         coinler_csv = pd.read_table(r"data.csv",sep=(";"))
#         a_coin_liste=coinler_csv
#         self.a_b_coinleri_listele.setText(str(a_coin_liste))
        
#     def a_coin_genel_istatistik_buton(self):
#         coinler_csv = pd.read_table(r"data.csv",sep=(";"))
#         a_coin_liste = coinler_csv.A_COİN
#         a_coin_genel_istatistik = a_coin_liste.describe()
#        # np.array(a_coini_1_aylik_liste).max()
#         #np.array(a_coini_1_aylik_liste).min()
#         self.a_coin_genel_istatistik.setText(str(a_coin_genel_istatistik))
#     def b_coin_genel_istatistik_buton(self):
#         coinler_csv = pd.read_table(r"data.csv",sep=(";"))
#         b_coin_liste = coinler_csv.B_COİN
#         b_coin_genel_istatistik = b_coin_liste.describe()
# #        np.array(b_coini_1_aylik_liste).max()
#  #       np.array(b_coini_1_aylik_liste).min()
#         self.b_coin_genel_istatistik.setText(str(b_coin_genel_istatistik))
#     def a_coin_grafik_buton(self):
#         self.grafik_a.setPixmap(QPixmap("Figure_1.jpeg"))
        
#     def b_coin_grafik_buton(self):
#         self.grafik_b.setPixmap(QPixmap("Figure_3.jpeg"))
    
#     def a_coin_5_gunluk_tahmin_buton(self):
#         coinler_csv = pd.read_table(r"data.csv",sep=(";"))
#         a_coini_1_aylik_sozluk = dict(coinler_csv.A_COİN[:])  
#         a_coini_1_aylik_liste = list(a_coini_1_aylik_sozluk.values())
#         a_coini_dict = {
#             "gunler":list(a_coini_1_aylik_sozluk.keys()),
#             "degerler":list(a_coini_1_aylik_sozluk.values())    
#             }
#         a_coini_df = pd.DataFrame(data = a_coini_dict)
#         gunler_a = a_coini_df[["gunler"]]
#         degerler_a = a_coini_df[["degerler"]]
#         x_train_a,x_test_a,y_train_a,y_test_a=train_test_split(gunler_a,degerler_a,test_size=0.33,random_state=0)
#         poly_reg = PolynomialFeatures(degree=2)
#         gunler_poly = poly_reg.fit_transform(gunler_a)
#         lin_reg2=LinearRegression()
#         lin_reg2.fit(gunler_poly,degerler_a)
#         linear_reg = LinearRegression()
#         linear_reg.fit(x_train_a,y_train_a)
#         tahmin_1_a = linear_reg.predict([[32]])
#         tahmin_2_a = linear_reg.predict([[33]])
#         tahmin_3_a = linear_reg.predict([[34]])
#         tahmin_4_a = linear_reg.predict([[35]])
#         tahmin_5_a = linear_reg.predict([[36]])
       
#         rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
#         rf_reg.fit(gunler_a,degerler_a)
#         karar_agaci_a_1 = rf_reg.predict([[32]])
#         karar_agaci_a_2 = rf_reg.predict([[33]])
#         karar_agaci_a_3 = rf_reg.predict([[34]])
#         karar_agaci_a_4 = rf_reg.predict([[35]])
#         karar_agaci_a_5 = rf_reg.predict([[36]])
        
#         polySonuc_1_a = lin_reg2.predict(poly_reg.fit_transform([[32]]))
#         polySonuc_2_a = lin_reg2.predict(poly_reg.fit_transform([[33]]))
#         polySonuc_3_a = lin_reg2.predict(poly_reg.fit_transform([[34]]))
#         polySonuc_4_a = lin_reg2.predict(poly_reg.fit_transform([[35]]))
#         polySonuc_5_a = lin_reg2.predict(poly_reg.fit_transform([[36]])) 
#         self.a_coin_tahmin_1.setText(str(f"1. Gün Tahmin Değeri: {polySonuc_1_a}"))
#         self.a_coin_tahmin_2.setText(str(f"2. Gün Tahmin Değeri: {polySonuc_2_a}"))
#         self.a_coin_tahmin_3.setText(str(f"3. Gün Tahmin Değeri: {polySonuc_3_a}"))
#         self.a_coin_tahmin_4.setText(str(f"4. Gün Tahmin Değeri: {polySonuc_4_a}"))
#         self.a_coin_tahmin_5.setText(str(f"5. Gün Tahmin Değeri: {polySonuc_5_a}"))
#     def b_coin_5_gunluk_tahmin_buton(self):
#         coinler_csv = pd.read_table(r"data.csv",sep=(";"))
#         b_coini_1_aylik_sozluk = dict(coinler_csv.B_COİN[:])
#         b_coini_1_aylik_liste = list(b_coini_1_aylik_sozluk.values())
#         b_coini_dict = {
#             "gunler":list(b_coini_1_aylik_sozluk.keys()),
#             "degerler":list(b_coini_1_aylik_sozluk.values())    
#             }
#         b_coini_df = pd.DataFrame(data = b_coini_dict)
#         gunler_b = b_coini_df[["gunler"]]
#         degerler_b = b_coini_df[["degerler"]]
#         x_train_b,x_test_b,y_train_b,y_test_b=train_test_split(gunler_b,degerler_b,test_size=0.33,random_state=0)
#         poly_reg = PolynomialFeatures(degree=2)
#         gunler_poly = poly_reg.fit_transform(gunler_b)
#         lin_reg2=LinearRegression()
#         lin_reg2.fit(gunler_poly,degerler_b)
#         linear_reg = LinearRegression()
#         linear_reg.fit(x_train_b,y_train_b)
#         tahmin_1_a = linear_reg.predict([[32]])
#         tahmin_2_a = linear_reg.predict([[33]])
#         tahmin_3_a = linear_reg.predict([[34]])
#         tahmin_4_a = linear_reg.predict([[35]])
#         tahmin_5_a = linear_reg.predict([[36]])
       
#         rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
#         rf_reg.fit(gunler_b,degerler_b)
#         karar_agaci_b_1 = rf_reg.predict([[32]])
#         karar_agaci_b_2 = rf_reg.predict([[33]])
#         karar_agaci_b_3 = rf_reg.predict([[34]])
#         karar_agaci_b_4 = rf_reg.predict([[35]])
#         karar_agaci_b_5 = rf_reg.predict([[36]])
        
#         polySonuc_1_b = lin_reg2.predict(poly_reg.fit_transform([[32]]))
#         polySonuc_2_b = lin_reg2.predict(poly_reg.fit_transform([[33]]))
#         polySonuc_3_b = lin_reg2.predict(poly_reg.fit_transform([[34]]))
#         polySonuc_4_b = lin_reg2.predict(poly_reg.fit_transform([[35]]))
#         polySonuc_5_b = lin_reg2.predict(poly_reg.fit_transform([[36]]))
#         self.b_coin_tahmin_1.setText(str(f"1. Gün Tahmin Değeri: {polySonuc_1_b}"))
#         self.b_coin_tahmin_2.setText(str(f"2. Gün Tahmin Değeri: {polySonuc_2_b}"))
#         self.b_coin_tahmin_3.setText(str(f"3. Gün Tahmin Değeri: {polySonuc_3_b}"))
#         self.b_coin_tahmin_4.setText(str(f"4. Gün Tahmin Değeri: {polySonuc_4_b}"))
#         self.b_coin_tahmin_5.setText(str(f"5. Gün Tahmin Değeri: {polySonuc_5_b}"))
#     def tahminleri_kaydet_buton(self):
#         polySonuc1_a = self.a_coin_tahmin_1.text()
#         polySonuc2_a = self.a_coin_tahmin_2.text()
#         polySonuc3_a = self.a_coin_tahmin_3.text()
#         polySonuc4_a = self.a_coin_tahmin_4.text()
#         polySonuc5_a = self.a_coin_tahmin_5.text()
#         polySonuc1_b = self.b_coin_tahmin_1.text()
#         polySonuc2_b = self.b_coin_tahmin_2.text()
#         polySonuc3_b = self.b_coin_tahmin_3.text()
#         polySonuc4_b = self.b_coin_tahmin_4.text()
#         polySonuc5_b = self.b_coin_tahmin_5.text()
        
#         baglanti=mysql.connector.connect(user="root",password="",host="127.0.0.1",database="coinler")
#         isaretci=baglanti.cursor()
#         isaretci.execute('''INSERT INTO coin_tahminleri(a_coin_tahmin_1,a_coin_tahmin_2,a_coin_tahmin_3,a_coin_tahmin_4,a_coin_tahmin_5,b_coin_tahmin_1,b_coin_tahmin_2,b_coin_tahmin_3,b_coin_tahmin_4,b_coin_tahmin_5) 
#                          VALUES ("%s","%s","%s","%s","%s","%s","%s","%s","%s","%s")'''%(polySonuc1_a,polySonuc2_a,polySonuc3_a,polySonuc4_a,polySonuc5_a,polySonuc1_b,polySonuc2_b,polySonuc3_b,polySonuc4_b,polySonuc5_b))
#         baglanti.commit()
#         baglanti.close

# uyg=QApplication([])
# pencere=coinler()
# pencere.setGeometry(100,100,1000,800)
# pencere.show()
# uyg.exec_()

df_2020 = pd.read_csv("eth-2020.csv") 
for i in df_2020.eth:
    print(i)
