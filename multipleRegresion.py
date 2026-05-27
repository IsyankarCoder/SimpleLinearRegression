import numpy as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from  matplotlib.cbook import boxplot_stats
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error,mean_absolute_error

# Veri okuma
veriSeti = pd.read_csv("heart.data.csv")
print(veriSeti.head(6))

# Veri Ön-İşleme
# Verisetinde ki ilk sütun analizlerde kullanılmayacağı için verisetinden çıkarılmıştır

veriSeti = veriSeti.iloc[:,1:4]
veriSeti = veriSeti.rename(columns={"biking":"bisikletKullanimi","smoking":"sigaraKullanimi","heart.disease":"kalpHastaligi"})
print(veriSeti.head(6))
print(veriSeti.describe())

# Bisikletle işe giden insanların yüzdesi (bisikletKullanimi)
# ve sigara içen insanların yüzdesi (sigaraDurum) nitelikleri
# ayrı ayrı kalp hastalığı olan insanların yüzdesi ile serpilme diyagramı üzerinde incelenmiştir.

sns.scatterplot(x="kalpHastaligi",y="bisikletKullanimi",data=veriSeti)
plt.show()

sns.scatterplot(x="kalpHastaligi",y="sigaraKullanimi",data=veriSeti)
plt.show()

# Ayrıca eş-doğrusallık (multi-colinearity) açısından sigara içen insanların yüzdesi (sigaraDurum) 
# ve bisikletle işe giden insanların yüzdesi (bisikletKullanimi) niteliklerinin kendi arasındaki ilişki de incelenmiştir. 
# Şekil 92’deki serpilme diyagramında sigara içen insanların yüzdesi (sigaraDurum) 
# ile bisikletle işe giden insanların yüzdesi (bisikletKullanimi) arasında
# pozitif ya da negatif yönde doğrusal bir ilişki tespit edilememiştir.
sns.scatterplot(x="bisikletKullanimi",y="sigaraKullanimi",data=veriSeti)
plt.show()

# Hedef nitelik olan kalp hastalığı olan insanların yüzdesi (kalpHastaligi)
# niteliğinin kutu grafiği yardımı ile incelenmiştir (Şekil 93). 
# Nitelikte herhangi bir aykırı değer tespit edilmemiştir.
# Kutu grafiğinden sonra verilen ekran görüntüsünde fliers bölümünde boş bir dizi yer aldığı görülebilir

sns.boxplot(x="kalpHastaligi",data=veriSeti,palette="summer")
print(boxplot_stats(veriSeti.kalpHastaligi))

plt.show()

# Verisetinde eksik değer kontrolü yapılmış herhangi bir eksik değere rastlanmamıştır
print(veriSeti.isnull().sum())

#Elde edilen sonuca göre; bisikletle işe giden insanların yüzdesi (bisikletKullanimi) ile kalp hastalığı olan insanların 
# yüzdesi (kalpHastaligi) arasında negatif yönde güçlü, anlamlı bir ilişkinin varlığından söz edilebilir 
# Pearson Katsayısı = -0.94 , p-degeri <0.05  3.54974454862926e−226
# Bisiklet kullanımı ile kalp hastalığı arasında çok güçlü,
# negatif yönlü ve istatistiksel olarak anlamlı bir ilişki vardır (r = -0.94, p < 0.001).

#p-değeri	Yorum
#p < 0.05	Anlamlı
#p < 0.01	Çok güçlü anlamlılık
#p > 0.05	Anlamlı değil

r, p =pearsonr(veriSeti.kalpHastaligi,veriSeti.bisikletKullanimi)
print("Korelasyon katsayisi:", r)
print("p-değeri:", p)

r1,p1 = pearsonr(veriSeti.kalpHastaligi,veriSeti.sigaraKullanimi)
print("Korelasyon katsayisi 1 :", r1)
print("p-değeri 1:", p1)

r2,p2= pearsonr(veriSeti.sigaraKullanimi,veriSeti.bisikletKullanimi)
print("Korelasyon katsayisi 2 :", r2)
print("p-değeri 2:", p2)

# Basit doğrusal regresyonda olduğu gibi hedef niteliğin normal (Gauss) 
# dağılımına uyup uymadığı ise histogram yardımı ile incelenebilir

sns.histplot(data=veriSeti,x="kalpHastaligi",color="hotpink")
plt.show()

# eğitim ve test veri setleri oluşturulmuştur. 
# Verinin %70’i eğitim (egitim), %30’u ise test veri setinde olacak şekilde (frac = 0.7) rastgele ikiye ayrılmıştır.

egitim = veriSeti.sample(frac=0.7,replace=False,random_state=1)
ind=veriSeti.index.isin(egitim.index)
test = veriSeti[~ind]

 # Modelleme
lr_model = smf.ols(formula="kalpHastaligi ~ bisikletKullanimi + sigaraKullanimi",data=egitim).fit()
print(lr_model.summary())
print(lr_model.params)
print(lr_model.params.index)


intercept = lr_model.params.iloc[0]
coef = lr_model.params.iloc[1]

#Regresyon Denklemi
# kalpHastaligi = 14.935 +(-0.200)* bisikletKullanimi+ 0.181 * sigaraKullanimi
print("Regresyon Modeli")
#print("Mutluluk = %.3f + %.3f * gelir " % (intercept,coef))

print(
    "kalpHastaligi = %.3f + (%.3f)* bisikletKullanimi + %.3f * sigaraKullanimi"
    % (
        lr_model.params["Intercept"],
        lr_model.params["bisikletKullanimi"],
        lr_model.params["sigaraKullanimi"]
    )
)

# R^2 değerine göre (Adj. R-squared) basit doğrusal regresyon 
# modelinin verideki toplam değişkenliğin %98’ini açıkladığı söylenebilir.  
r_sq= lr_model.rsquared
print("Modelim R^2 degeri = %.2f" %r_sq)

# Örneğin test veri setindeki ilk örneğin bisikletKullanimi niteliği 65.13 ve sigaraDurum niteliği 2.22’dir.
# Bu değerler çoklu doğrusal regresyon denkleminde yerine yazılırsa, 
# test veri setindeki ilk örneğin kalp hastalığı olan insanların yüzdesi değeri 2.33 olarak hesaplanır.
kalpHastaligi_test1 =  lr_model.params["Intercept"]  + lr_model.params["bisikletKullanimi"] * test.loc[1,"bisikletKullanimi"] + lr_model.params["sigaraKullanimi"] * test.loc[1,"sigaraKullanimi"]
print(kalpHastaligi_test1)

lr_tahminler = lr_model.predict(test[["bisikletKullanimi","sigaraKullanimi"]])
print("LR Model Tahminleri : ",lr_tahminler.head(5),sep="\n")
print("Gerçek Değerler: ",test.kalpHastaligi.head(5),sep="\n")

performansVeri = pd.DataFrame(np.column_stack([test.kalpHastaligi,lr_tahminler]))
performansVeri.columns=["y_test","lr_tahminler"]

print(performansVeri)

# Hata
performansVeri["error"] = lr_tahminler-test.kalpHastaligi

error = performansVeri["error"]

ME = np.mean(error)
MAE = np.mean(np.abs(error))

eps = 1e-10
MPE = np.mean(error / (performansVeri["y_test"] + eps)) * 100
MAPE = np.mean(np.abs(error / (performansVeri["y_test"] + eps))) * 100

MSE = np.mean(error ** 2)
RMSE = np.sqrt(MSE)

print("ME = ",ME)
print("MAE = ",MAE)
print("MPE = ",MPE)
print("MAPE = ",MAPE)
print("MSE = ",MSE)
print("RMSE = ",RMSE)