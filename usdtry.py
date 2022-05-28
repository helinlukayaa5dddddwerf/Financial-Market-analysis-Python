#kutuphaneleri yukleme
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

#veri yukleme
veriler = pd.read_csv("usdtry.csv")

#data frame dilimleme
X = veriler['gun']
Y = veriler['kapanis_fiyat']

#string fiyat verisinin float veriye dönüştürülmesi
y1 = [int(w.replace('.', '')) / 1000000 for w in Y]

#numpy array donusumu
x = np.array(X.values)
y = np.array(y1)

#verilerin dagilimi
plt.scatter(x, y, color="orange")
plt.xlabel("Gün") 
plt.ylabel("Fiyat")
plt.title("Verilerin Dağılımı")
plt.show()

#lineer regresyon
x = x.reshape((-1, 1))
linreg = LinearRegression().fit(x, y)

#lineer regresyon gorsellestirme
plt.scatter(x, y, color="orange")
plt.plot(x, linreg.predict(x), color ="green")
plt.xlabel("Gün") 
plt.ylabel("Fiyat")
plt.title("Lineer Regresyon")
plt.show()

#lineer tahmin
print("Lineer Regresyon Tahmin:", linreg.predict([[280]]))

#polinomsal regresyon
degree= [2,3,4,5,6,7,8,9,10,11]
performance = []
for i in degree:
    polreg = PolynomialFeatures(degree=i)
    xpol = polreg.fit_transform(x)
    linreg = LinearRegression().fit(xpol, y)
    print(str(i) + 'dereceden polinom için r2 skoru:' ,r2_score(y, linreg.predict(polreg.fit_transform(x))))
    performance.append(r2_score(y, linreg.predict(polreg.fit_transform(x))))

    #polinomsal regresyon gorsellestirme
    plt.scatter(x, y, color ="orange")
    plt.plot(x, linreg.predict(polreg.fit_transform(x)), color = "green")
    plt.xlabel("Gün") 
    plt.ylabel("Fiyat")
    plt.title("Polinomsal Regresyon (Derece:"+ str(i) +")")
    plt.show()
    
# En iyi sonuç elde edilen polinomsal regresyon (r2 score'u 1'e en yakın olandır.)
for j in range (len(performance)):   
    if performance[j] == max(performance):
        polreg = PolynomialFeatures(degree = j+2) #index num
        xpol = polreg.fit_transform(x)
        linreg = LinearRegression().fit(xpol, y)
        performance.append(r2_score(y, linreg.predict(polreg.fit_transform(x))))
        
        #polinomsal regresyon gorsellestirme
        plt.scatter(x, y, color ="orange")
        plt.plot(x, linreg.predict(polreg.fit_transform(x)), color = "green")
        plt.xlabel("Gün") 
        plt.ylabel("Fiyat")
        plt.title("Polinomsal Regresyon (Derece:"+ str(j+2) +")")
        plt.show()
            
    else:
        continue

#polinomsal tahmin
print("Polinomsal Regresyon Tahmin:",linreg.predict(polreg.fit_transform([[258]])))
