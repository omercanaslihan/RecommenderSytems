###Association Rule Based Recommender System###
##İş Problemi##
#Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
#Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca ulaşılmasını sağlamaktadır.
#Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak Association Rule Learning ile
#ürün tavsiye sistemi oluşturulmak istenmektedir.

#Veri Seti Hikayesi#
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır. Alınan her hizmetin tarih ve saat bilgisini içermektedir.

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#Görev 1: Veriyi Hazırlama
#Adım 1: armut_data.csv dosyasını okutunuz.
df_ = pd.read_csv("armut_data.csv")
df = df_.copy()
df.head()

#Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir. ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek
#yeni bir değişken oluşturunuz.
df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

#Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır. Association Rule Learning uygulayabilmek
#için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in
#8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti; 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile
#tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
#ile birleştirirek ID adında yeni bir değişkene atayınız.
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["Date"] = df["CreateDate"].dt.strftime("%Y-%m")
df["ID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]

#Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
#Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.
invoice_product_df = df.groupby(["ID", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x:1 if x > 0 else 0)
invoice_product_df.head()

#Adım 2: Birliktelik kurallarını oluşturunuz.
frequent_items = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_items, metric="support", min_threshold=0.01)
rules.head()

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(rules, "2_0", 2)
arl_recommender(rules, "2_0", 3)