import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,accuracy_score,recall_score

# Veri Okuma
veriseti = pd.read_csv("dataR2.csv")

# Veri Önİşleme
veriseti = veriseti.rename(columns={"Classification":"Karar"})
print(veriseti["Karar"].value_counts())
veriseti["Karar"]=np.where(veriseti["Karar"]==1,"Saglikli","Kanser")

print(veriseti["Karar"].value_counts())

veriseti.Karar = veriseti.Karar.astype("category")
pd.set_option("display.max_columns",20)
veriseti.describe(include='all')

print(veriseti.dtypes)

# Veri setinde boş olan kayıt var mı
print(veriseti.isnull().sum())

# Hedef nitelik dışında ki diğer tüm nitelikleri içeren korelasyon ısı haritası
my_cors = pd.DataFrame(np.corrcoef(veriseti.iloc[:,0:9],rowvar=False).round(2),columns=veriseti.columns[0:9])
my_cors.index=veriseti.columns[0:9]

# Korelasyon Isı Haritasi En yüksek Pearson korelasyon katsayısının 
# HOMA ve Insulin nitelikleri arasında olduğu tespit edilmiştir.
# HOMA ve Insulin nitelikleri arasında pozitif yönde güçlü doğrusal bir ilişkinin varlığından söz edilebilir. 
# Detaylı korelasyon incelemeleri p-değerlerinin de hesaplanması ile gerçekleştirilebilir ve yorumlanabilir.
sns.heatmap(my_cors,annot=True,square=True,cmap=sns.color_palette("flare",as_cmap=True))

plt.show()

# Scikit-learn kütüphanesinin model_selection modülünden train_test_split() 
# fonksiyonu kullanılarak eğitim ve test veri setleri oluşturulmuştur. 
# Verinin %70’i eğitim (egitim), %30’u ise test veri setinde olacak 
# şekilde (frac = 0.7) rastgele ayrılmıştır.

X_train,X_test,y_train,y_test = train_test_split(veriseti.iloc[:,0:9],veriseti.iloc[:,9],test_size=0.3,random_state=1)

# sklearn.preprocessing modülünden MinMaxScaler minimum-maksimum normalizasyon yöntemi ile veri setleri normalize edilmiştir.
# Bunun için öncelikle scaler.fit_transform() fonksiyonu X_train’e uygulanmış ve 
# eğitim veri setindeki tüm nitelikler 0 ile 1 arasında olacak şekilde ayarlanmıştır. 
# Bu aşamada test veri seti normalize edilirken scaler nesnesinin 
# eğitim veri setinden öğrendiği parametreler scaler.transform() fonksiyonu yardımı ile X_test’e uygulanmıştır.
# Bu uygulamanın sebebi, test verilerinin modelin daha önce görmediği yeni veriler olduğunun varsayılmasıdır
# Normalize edilen eğitim ve test veri setleri X_train_n ve X_test_n olarak saklanmıştır.

scaler = MinMaxScaler()
X_train_n= pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test_n=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

print(X_train_n.describe())

# Modelleme
# k-NN modelinin (knn_modeli) oluşturulabilmesi için KNeighborsClassifier() kullanılmıştır. 
# Bunun için fonksiyonun ilk argümanı olarak k komşu sayısı (n_neighbors) 5 verilmiştir.
# İkinci argüman olarak ise uzaklık fonksiyonu (metric) euclidean (Öklid) seçilmiştir.
# Ardından knn_modeli modelinin 
# fit() fonksiyonuna normalize edilen veri seti ve eğitim veri setinin hedef niteliği verilmiştir.

knn_modeli = KNeighborsClassifier(n_neighbors=5,metric="euclidean")
knn_modeli.fit(X_train_n,y_train)

# Performans Değerlendirme
# Test veri setindeki örnekler için oluşturulan k-NN modeli (knn_modeli) tahminlerinin bulunabilmesi için 
# lr_model.predict()
# fonksiyonundan yararlanılabilir. Fonksiyona test veri setindeki tahmini sağlayan nitelikler (X_test_n) verilir.

y_tahmin = knn_modeli.predict(X_test_n)

# Modelin test verisetinde ki tüm örnekler için mutluluk puanı tahminleri y_tahmin dizisinde tutulmaktadır.
# Test veri setinde bulunan tüm örnekler için k-NN modelinin tahmin ettiği ilk 5 değer (y_tahmin)
# ve test veri setindeki gerçek meme kanseri kararına ilişkin değerlerin (y_test) 
# ilk 5 değeri aşağıdaki kod yardımı ile ekrana yazdırılmıştır.

print("k-NN Modeli Tahminleri : ",y_tahmin[0:5])
print("Gerçek Değerler : ",np.array(y_test[0:5]))

# y_tahmin ve y_test yardımı ile kontenjans tablosu oluşturulmuş ve bu tablo kullanılarak
# doğruluk, duyarlılık, belirleyicilik gibi çeşitli performans değerlendirme ölçütleri hesaplanmıştır. 
# “Kanser” sınıfı pozitif sınıf olarak seçilmiş olsun. 
# y_tahmin ve y_test’in ilk iki elemanı şu şekilde yorumlanabilir:

# • Gerçekte “Kanser”, k-NN modeli de “Kanser” olarak doğru tahmin etmiş,
# dolayısıyla bu bir TP’dir (true positive/doğru pozitif).

# • Gerçekte “Sağlıklı”, k-NN modeli ise “Kanser” olarak yanlış tahmin etmiş,
# dolayısıyla bu bir FP’dir (false positive/yanlış pozitif).

# Kontenjans tablosu oluşturmak için en basit yöntem olarak sklearn.metrics
# modülünden confusion_matrix() fonksiyonu kullanılabilir. 
# y_true parametresine test veri setindeki meme kanseri kararını gösteren gerçek değerler
# (y_test), y_pred parametresine ise k-NN modeli tahminleri (y_tahmin) verilmiştir. 
# labels parametresine ise veri setinin sınıf değerleri (kategorileri) olması gereken sırada verilmiştir. 
# Eğer labels parametresine herhangi bir değer verilmezse sınıf kategorileri 
# alfabetik sıraya göre yazdırılacaktır

my_cm = confusion_matrix(y_true=y_test,y_pred=y_tahmin,labels=["Saglikli","Kanser"])
print(my_cm)

# Python’da bu yolla elde edilen kontenjans tablosunda sütunlar tahmin değerlerini göstermekteyken 
# satırlar gerçek değerleri göstermektedir. 
# Test veri setinde 23 Kanser ve 12 Sağlıklı kategorisine ait örnek yer almaktadır.
# Sırasıyla; tn=7 (true negatives), fp=5 (false positives), fn=4 (false negatives), tp=19 (true positives) olarak elde edilmiştir.

# Kontenjans tablosunun görsel açıdan daha iyi çıktılanması için yine 
# sklearn.metrics içinden ConfusionMatrixDisplay() fonksiyonu kullanılabilir.
# Bu fonksiyona bir önceki aşamada oluşturulan kontenjans tablosu 
# my_conf ya da 
# bunun yerine y_true = y_test, y_pred = y_tahmin şeklinde parametreler verilebilir.
# display_labels parametresi de confusion_matrix() fonksiyonunun 
# labels parametresiyle benzer biçimde yorumlanabilir.

my_cm_p= ConfusionMatrixDisplay(my_cm,display_labels=["Saglikli","Kanser"])
my_cm_p.plot()
plt.show()

# Kontenjans tablosundaki tn, tp, fn, fp değerlerinin sırasıyla 
# aynı isimlerde tanımlanan değişkenlere atanabilmesi için my_cm.ravel() kullanılabilir.

tn,fp,fn,tp = my_cm.ravel()
print("True Negatives : ",tn)
print("False Positives : ",fp)
print("False Negatives : ",fn)
print("True Positives : ",tp)

# Dogruluk (accuracy)
dogruluk = (tp+tn)/(tp+tn+fp+fn)
# Hata oranı 
hata = 1- dogruluk
# Duyarlilik (sensivity)
duyarlilik = tp/(tp+fn)
# Belirleyicilik (Specifity)
belirleyicilik = tn/(tn+fp)





