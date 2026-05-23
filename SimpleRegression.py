import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.cbook import boxplot_stats
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error

#veri okuma
veriseti = pd.read_csv("income.data.csv")
print(veriseti.head(6))

#veri ön işleme
#veri setinde ki ilk sütun analizlerde kullanılmayacağı için veri setinden çıkarılmıştır

veriseti = veriseti.iloc[:,1:3]

#veri setinde ki niteliklerin sırasıyla gelir ve mutluluk olacak şekilde değiştirilmiştir.

veriseti =veriseti.rename(columns={"income":"gelir","happiness":"mutluluk"})
print(veriseti.describe())

#Veri setinde eksik değer kontrolü yapılmış; ancak herhangi bir eksik değere rastlanmamıştır.
print(veriseti.isnull().sum())


#Bunun için scipy kütüphanesinin stats modülünden pearsonr() fonksiyonu kullanılmıştır   
# Önceki bölümlerde korelasyon katsayısını hesaplamak için kullanılan fonksiyonlardan farkı, 
# Pearson korelasyon katsayısı (statistic) ile iki nitelik arasındaki ilişkinin anlam düzeyini içeren   -değerini (pvalue) de 
# hesaplıyor olmasıdır. İki nitelik arasındaki ilişkinin anlamlı olabilmesi için   -değerinin genellikle
# 0.05’ten küçük olması beklenir. Elde edilen sonuca göre gelir ve mutluluk arasındaki Pearson korelasyon katsayısı 0.87’dir. Hesaplanan   -değeri bilimsel gösterim (scientific notation) biçiminde verilmiştir. Yani; 3.956245289952218e-151 değeri    şeklinde ifade edilmektedir. Buna göre elde edilen p-değeri sıfıra oldukça yakın bir değerdir. 
# Dolayısıyla gelir ile mutluluk puanı arasında pozitif yönde güçlü, anlamlı bir ilişkinin varlığından söz edilebilir  
print(pearsonr(veriseti.gelir,veriseti.mutluluk))

sns.scatterplot(data=veriseti,x="gelir",y="mutluluk")
plt.show()

#Nitelikte herhangi bir aykırı değer tespit edilememiştir. 
#Kutu grafiğinden sonra gelen ekran görüntüsünde fliers bölümünde boş bir dizi yer aldığı görülebilir.
sns.boxplot(x="gelir",data=veriseti,palette="summer")
boxplot_stats(veriseti.gelir)

plt.show()

sns.histplot(data=veriseti, x="gelir",color="green")
plt.show()

# %70 egitim , %30 test
egitim = veriseti.sample(frac=0.7,replace=False,random_state=1)
ind= veriseti.index.isin(egitim.index)
test = veriseti[~ind]

#Doğrusal regresyon modelinin (lr_model) oluşturulabilmesi için smf.ols() fonksiyonu kullanılmıştır. 
#Bunun için fonksiyonun ilk argümanı olarak mutluluk ~ gelir biçiminde bağımlı değişkene karşı bağımsız değişken 
#olacak biçimde modelin formül gösterimi yer almaktadır. Eğitim veri seti ile model kurulacağından, 
#ikinci argüman olarak eğitim veri seti (egitim) verilmiştir. .fit() ile model oluşturulmuştur.
lr_model = smf.ols(formula="mutluluk ~ gelir",data=egitim).fit()

# regresyon modeli raporunda coef olarak verilenler model sabiti (Intercept) ve gelir değişkeni katsayısıdır (gelir). 
# P > |t| alanında listelenen değerler ise modele giren niteliklere ait anlamlılık düzeylerini göstermektedir.    
# p değeri çok düşük olduğu için (p<0.0001), sıfır hipotezi reddedebilir ve gelirin mutluluk üzerinde istatistiksel olarak anlamlı bir etkisi olduğu sonucuna varılabilir.
# Burada gelir ile mutluluk arasında anlamlı pozitif bir ilişki vardır 
# ve gelirdeki her bir birimlik artışa karşılık mutlulukta 0.7083 birimlik bir artış söz konusudur.
# Regresyon modelinin sabiti ve gelir niteliğine ait beta katsayılarına ayrıca ulaşmak için aşağıdaki 
# kod satırı kullanılabilir.
print(lr_model.params)
print(lr_model.summary())

print("Regresyon Modeli")
print("mutluluk = %.3f + %.3f * gelir" % (lr_model.params["Intercept"],lr_model.params["gelir"]))

# Regresyon Modeli tarafından açıklanan toplam değişkenliğin oranı R2(rekare) ile değerlendirilebilmektedir. 
# Bu değerin 1’e yakın olması, modelin veriye iyi uyum sağladığını,
# 0’a yakın olması ise zayıf biçimde uyum sağladığını göstermektedir. 
# lr_model.rsquared   --> 0.749
r_sq= lr_model.rsquared  
print("Modelin R^2 değeri = %.2f "% r_sq)

# mutluluk = 0.228 + 0.708 * gelir ,   lr_model.predict() fonksiyonundan yararlanılabilir. 
# Fonksiyona test veri setindeki bağımsız değişkenler (bu örnekte gelir niteliği) verilir.
# Modelin test veri setindeki tüm örnekler için mutluluk puanı tahminleri lr_tahminler dizisinde tutulmaktadır
# Tahmin değerleri ve gerçek değerlerden ilk 5 tanesi yazdırılırsa
# modelin gerçek değerlere ne kadar yakın ya da uzak tahminde bulunduğu incelenebilecektir. 
# Örneğin test veri setindeki ilk örneğin gerçek mutluluk puanı (mutluluk) 3.43’dir; 
# lr_model.predict() fonksiyonu ile de bu değer 3.75 olarak tahmin edilmiştir (lr_tahminler.iloc[0,])
# (hesaplarda sayıların virgülden sonra kullanılan basamak sayısına göre oldukça küçük bir farklılık çıkmıştır).
lr_tahminler = lr_model.predict(test[["gelir"]])

print("LR Modeli Tahminleri :",lr_tahminler.head(5),sep="\n")
print("Gerçek Değerler : ",test.mutluluk.head(5),sep="\n")

performansverisi = pd.DataFrame(np.column_stack([test.mutluluk,lr_tahminler]))
performansverisi.columns=["y_test","lr_tahminler"]
print(performansverisi.describe())

# Hata 
#  DataFrame nesnesindeki gerçek değerler ve tahmin edilen değerler kullanılarak öncelikle hata değerleri, 
#  sonra ise hata değerleri kullanılarak hesaplanabilen
#  ortalama hata (ME), 
#  ortalama mutlak hata (MAE),
#  ortalama yüzde hata (MPE), 
#  ortalama mutlak yüzde hata (MAPE),
#  ortalama karesel hata (MSE)
#  ve ortalama karesel hatanın karekökü (RMSE) hesaplanmıştır (Not: MSE, RMSE’nin karesidir).

performansverisi["error"] =lr_tahminler-test.mutluluk
print(performansverisi)
print(len(performansverisi))

ME = np.sum(performansverisi.error)/len(performansverisi)
MAE= np.sum(np.abs(performansverisi.error))/len(performansverisi)
MPE= 100* (np.sum(performansverisi.error/performansverisi.y_test))/len(performansverisi)
MAPE = 100*(np.sum(np.abs(performansverisi.error/performansverisi.y_test)))//len(performansverisi)
MSE = np.sum(pow(performansverisi.error,2))/len(performansverisi)
RMSE = np.sqrt(np.sum(pow(performansverisi.error,2)))/len(performansverisi)

print("ME =", ME)
print("MAE =",MAE)
print("MPE =",MPE)
print("MAPE =",MAPE)
print("MSE =",MSE)
print("RMSE =",RMSE)

# MAE, MSE ve RME sklearn.metrics içinde ki hazır fonksiyonlar kullanılarak da hesaplanabilir
mae = mean_absolute_error(y_true=test.mutluluk,y_pred=lr_tahminler)
mse =mean_squared_error(y_true=test.mutluluk,y_pred=lr_tahminler)
#rmse = mean_squared_error(y_true=test.mutluluk,y_pred=lr_tahminler,squared=False)

print(mae)
print(mse)
#print(rmse)

# katsayı 0.708’dir ve 0.228 değeri (Intercept) x = 0 
plt.scatter(test.gelir,test.mutluluk,color="hotpink")
plt.plot(test.gelir,lr_tahminler,color="navy",linewidth=3)
plt.show()
