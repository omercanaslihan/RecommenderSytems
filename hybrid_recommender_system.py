####Hybrid Recommender System#####
###İş Problemi###
#ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak 10 film önerisi yapınız.#

#Veri Seti Hikayesi#
#Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte
#bu filmlere yapılan derecelendirme puanlarını barındırmaktadır. 27.278 filmde 2.000.0263 derecelendirme içermektedir.
#Bu veri seti ise 17 Ekim 2016 tarihinde oluşturulmuştur. 138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri
#arasında verileri içermektedir. Kullanıcılar rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option('display.expand_frame_repr', False)

#Görev 1: Veri Hazırlama
#Adım1: movie,ratingverisetleriniokutunuz.
movie_df = pd.read_csv("movie.csv")
rating_df = pd.read_csv("rating.csv")
movie_df.head()
rating_df.head()
#Adım 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
df = pd.merge(rating_df, movie_df, on="movieId", how="left")
df.head()
#Adım 3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.head()
#Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz.
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values='rating')
user_movie_df.head()
# #Adım 5: Yapılan tüm işlemleri fonksiyonlaştırınız.
def prep_user_movie_df():
    import pandas as pd
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 500)
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    movie_df = pd.read_csv("movie.csv")
    rating_df = pd.read_csv("rating.csv")
    df = pd.merge(rating_df, movie_df, on="movieId", how="left")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values='rating')
    return  user_movie_df

#Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#Adım1: Rastgelebirkullanıcıid’siseçiniz.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
random_user
#Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
#Adım 3: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

#Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
#Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturunuz.
movies_watched_df = user_movie_df[movies_watched]
#Adım 2: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userid", "movie_count"]
user_movie_count.head()
#Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden users_same_movies adında bir liste oluşturunuz.
perc = len(movies_watched) * 60  / 100
user_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userid"]
user_same_movies.shape

#Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
final_df = movies_watched_df[movies_watched_df.index.isin(user_same_movies)]
final_df.head()
#Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["userid_1", "userid_2"]
corr_df = corr_df.reset_index()
corr_df.head()
#Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_df[(corr_df["userid_1"] == random_user) & (corr_df["corr"] > 0.65)][["userid_2", "corr"]]
top_users.head()
top_users.columns = ["userId", "corr"]
#Adım 4: top_users dataframe’ine rating veri seti ile merge ediniz.
top_users_score = top_users.merge(rating_df[["userId", "movieId", "rating"]], how="inner")
top_users_score.shape

#Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#Adım1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_score["weighted_reting"] = top_users_score["corr"] * top_users_score["rating"]
top_users_score.head()
#Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
#dataframe oluşturunuz.
recommendation_df = top_users_score.groupby("movieId").agg({"weighted_reting": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()
#Adım 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
movies_to_be_recommended = recommendation_df[recommendation_df["weighted_reting"] > 3.5].sort_values("weighted_reting", ascending=False)
movies_to_be_recommended.head()
#Adım 4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.
movies_to_be_recommended.merge(movie_df[["movieId", "title"]])["title"]
#1                                       Lamerica (1994)
#2                               To Live (Huozhe) (1994)
#3                                 Talk of Angels (1998)
#4                                    October Sky (1999)
#5     Twice Upon a Yesterday (a.k.a. Man with Rain i..

##Item Based Recommendation##
#Görev 1: Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.
#Adım1: movie,rating veri setlerini okutunuz.
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
movie_df = pd.read_csv("movie.csv")
rating_df = pd.read_csv("rating.csv")
#Adım 2: Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
movie_id = rating_df[(rating_df["userId"] == random_user) & (rating_df["rating"] == 5.0)].sort_values("timestamp", ascending=False)["movieId"][0:1].values[0]
#Adım 3: User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
movie_df = user_movie_df[movie_df[movie_df["movieId"] == movie_id]["title"].values[0]]
#Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(20)
#Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
item_based_recommendations = user_movie_df.corrwith(movie_df).sort_values(ascending=False).index[1:6]