import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import geopandas as gpd
import contextily as ctx

from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.sidebar.markdown("### Preprocesare Date")
metoda_extreme = st.sidebar.selectbox("Tratament valori extreme",
                                      ["Nicio acțiune", "Eliminare IQR", "Winsorizare", "Transformare logaritmică"])
aplica_binning_timp = st.sidebar.checkbox("Aplică Binning pentru timp", value=True)
aplica_codificare_ciclică = st.sidebar.checkbox("Aplică codificare ciclică pentru ore și zile", value=True)
aplica_indicator_evenimente = st.sidebar.checkbox("Aplică indicator evenimente speciale", value=True)

st.sidebar.markdown("### Scalare Caracteristici")
metoda_scalare = st.sidebar.selectbox("Alege metoda de scalare",
                                      ["Nicio scalare", "MinMax", "Standard", "Robust", "MaxAbs"])
coloane_scalare = st.sidebar.multiselect("Coloane de scalat",
                                         ["fare_amount", "passenger_count", "pickup_longitude", "pickup_latitude",
                                          "dropoff_longitude", "dropoff_latitude"])


@st.cache_data
def incarca_date(metoda_extreme, aplica_binning_timp, aplica_codificare_ciclică, aplica_indicator_evenimente,
                 metoda_scalare, coloane_scalare):
    df = pd.read_csv("uber.csv", parse_dates=['pickup_datetime'])

    st.subheader("Valori lipsă înainte de prelucrare")
    st.write(df.isnull().sum()[df.isnull().sum() > 0])
    df.dropna(inplace=True)

    def calculeaza_iqr(col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    coloane_verificare = ["fare_amount", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude",
                          "dropoff_latitude"]

    if metoda_extreme == "Eliminare IQR":
        total_outlieri = 0
        nr_total_valori = df.shape[0] * len(coloane_verificare)
        for col in coloane_verificare:
            lower, upper = calculeaza_iqr(col)
            outlieri = df[(df[col] < lower) | (df[col] > upper)].shape[0]
            total_outlieri += outlieri
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        st.markdown(f"**Eliminare IQR:** S-au identificat {total_outlieri} outlieri din {nr_total_valori} valori.")
    elif metoda_extreme == "Winsorizare":
        total_inlocuite = 0
        for col in coloane_verificare:
            lower, upper = calculeaza_iqr(col)
            inlocuite = df[(df[col] < lower) | (df[col] > upper)].shape[0]
            total_inlocuite += inlocuite
            df[col] = df[col].clip(lower, upper)
        st.markdown(f"**Winsorizare:** S-au înlocuit {total_inlocuite} valori extreme.")
    elif metoda_extreme == "Transformare logaritmică":
        df = df[df["fare_amount"] > 0.01]
        df["fare_amount"] = np.log1p(df["fare_amount"])
        st.markdown("**Transformare logaritmică:** S-a aplicat transformarea log1p pe fare_amount.")
    else:
        st.markdown("**Niciun tratament pentru valori extreme aplicat.**")

    df["ora"] = df["pickup_datetime"].dt.hour
    df["zi_saptamana"] = df["pickup_datetime"].dt.dayofweek
    df["an"] = df["pickup_datetime"].dt.year

    if aplica_binning_timp:
        bins = [0, 6, 12, 18, 24]
        etichete = ["Noapte", "Dimineață", "Prânz", "Seară"]
        df["categorie_timp"] = pd.cut(df["ora"], bins=bins, right=False, labels=etichete)
    else:
        df["categorie_timp"] = df["ora"]

    if aplica_codificare_ciclică:
        df["ora_sin"] = np.sin(2 * np.pi * df["ora"] / 24)
        df["ora_cos"] = np.cos(2 * np.pi * df["ora"] / 24)
        df["zi_sin"] = np.sin(2 * np.pi * df["zi_saptamana"] / 7)
        df["zi_cos"] = np.cos(2 * np.pi * df["zi_saptamana"] / 7)

    if aplica_indicator_evenimente:
        evenimente = {
            "Crăciun": (12, 25),
            "Halloween": (10, 31),
            "Anul Nou": (1, 1),
            "Paște": (4, 16)
        }

        def verifica_eveniment(dt):
            for nume, (luna, zi) in evenimente.items():
                if dt.month == luna and dt.day == zi:
                    return nume
            return "Fără eveniment"

        df["eveniment_special"] = df["pickup_datetime"].apply(verifica_eveniment)

    if metoda_scalare != "Nicio scalare" and coloane_scalare:
        if metoda_scalare == "MinMax":
            scaler = MinMaxScaler()
        elif metoda_scalare == "Standard":
            scaler = StandardScaler()
        elif metoda_scalare == "Robust":
            scaler = RobustScaler()
        elif metoda_scalare == "MaxAbs":
            scaler = MaxAbsScaler()
        df[coloane_scalare] = scaler.fit_transform(df[coloane_scalare])

    def haversine(lon1, lat1, lon2, lat2):
        R = 6371
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    df["distanta_km"] = haversine(df["pickup_longitude"], df["pickup_latitude"], df["dropoff_longitude"],
                                  df["dropoff_latitude"])

    df["tarif_per_km"] = df["fare_amount"] / (df["distanta_km"] + 1e-6)
    media_tarif = df["tarif_per_km"].mean()
    std_tarif = df["tarif_per_km"].std()
    prag = media_tarif + 2 * std_tarif
    df["outlier_tarif"] = df["tarif_per_km"] > prag

    if aplica_binning_timp:
        agregare_timp = df.groupby("categorie_timp")["fare_amount"].mean().reset_index()
    else:
        agregare_timp = df.groupby("ora")["fare_amount"].mean().reset_index()
    agregare_zi = df.groupby("zi_saptamana")["fare_amount"].mean().reset_index()
    agregare_an = df.groupby("an")["distanta_km"].mean().reset_index()
    agregare_pasageri = df.groupby("passenger_count")["fare_amount"].mean().reset_index()

    return df, agregare_timp, agregare_zi, agregare_an, agregare_pasageri


df, agregare_timp, agregare_zi, agregare_an, agregare_pasageri = incarca_date(metoda_extreme, aplica_binning_timp,
                                                                              aplica_codificare_ciclică,
                                                                              aplica_indicator_evenimente,
                                                                              metoda_scalare, coloane_scalare)

st.title("Analiza Tarifelor Uber")
st.markdown(
    "Această aplicație prelucrează și analizează datele curselor Uber utilizând Streamlit, pandas și diverse tehnici de prelucrare a datelor.")

st.subheader("Preview Date")
st.write(df.head())

st.subheader("Statistici Descriptive")
st.write(df.describe())

st.subheader("Distribuția Tarifelor")
fig, ax = plt.subplots()
sns.histplot(df["fare_amount"], bins=50, kde=True, ax=ax)
ax.set_xlabel("Tarif")
ax.set_ylabel("Frecvență")
st.pyplot(fig)
st.markdown(
    "**Interpretare:** Majoritatea curselor au tarife mici intre 5-20 USD. Distributia este asimetrica cu valori extreme peste 50 USD, probabil curse lungi sau la ore de varf.")

st.subheader("Tendințe în Funcție de Timp")
fig, ax = plt.subplots()
sns.boxplot(x=df["categorie_timp"], y=df["fare_amount"], ax=ax)
ax.set_xlabel("Categorie de timp")
ax.set_ylabel("Tarif")
st.pyplot(fig)
st.markdown(
    "**Interpretare:** Tarifele sunt mai mari noaptea si seara. Orele de varf probabil duc la tarife mai ridicate datorita cererii crescute.")

st.subheader("Relația dintre Numărul de Pasageri și Tarif")
fig, ax = plt.subplots()
sns.boxplot(x=df["passenger_count"], y=df["fare_amount"], ax=ax)
ax.set_xlabel("Număr de pasageri")
ax.set_ylabel("Tarif")
st.pyplot(fig)
st.markdown(
    "**Interpretare:** Nu exista o corelatie clara intre numarul de pasageri si tarif. Cursele cu 1-2 pasageri sunt predominante.")

st.subheader("Relația dintre Distanță și Tarif")
fig, ax = plt.subplots()
sns.scatterplot(x=df["distanta_km"], y=df["fare_amount"], hue=df["outlier_tarif"], palette="coolwarm", ax=ax)
ax.set_xlabel("Distanță (km)")
ax.set_ylabel("Tarif")
st.pyplot(fig)
st.markdown(
    "**Interpretare:** Relatie aproximativ liniara intre distanta si tarif. Valorile extreme (rosii) reprezinta probabil curse cu tarife nejustificate pentru distanta parcursa.")

st.subheader("Tarif Mediu pe Zi a Săptămânii")
fig, ax = plt.subplots()
sns.barplot(x=agregare_zi["zi_saptamana"], y=agregare_zi["fare_amount"], ax=ax)
ax.set_xlabel("Zi a săptămânii (0 = Luni, 6 = Duminică)")
ax.set_ylabel("Tarif mediu")
st.pyplot(fig)
st.markdown(
    "**Interpretare:** Tarifele sunt putin mai mari in mijlocul saptamanii.")

st.subheader("Distanța Medie pe An")
fig, ax = plt.subplots()
sns.lineplot(x=agregare_an["an"], y=agregare_an["distanta_km"], marker='o', ax=ax)
ax.set_xlabel("An")
ax.set_ylabel("Distanță medie (km)")
st.pyplot(fig)
st.markdown(
    "**Interpretare:** Distanta medie a curselor scade usor in timp, posibil datorita cresterii numarului de curse scurte in oras.")

st.subheader("Tarif Mediu în Funcție de Numărul de Pasageri")
fig, ax = plt.subplots()
sns.barplot(x=agregare_pasageri["passenger_count"], y=agregare_pasageri["fare_amount"], ax=ax)
ax.set_xlabel("Număr de pasageri")
ax.set_ylabel("Tarif mediu")
st.pyplot(fig)
st.markdown(
    "**Interpretare:** Cursele cu mai mult de un pasager tind sa fie putin mai scumpe.")

if aplica_indicator_evenimente:
    st.subheader("Tarif Mediu în Funcție de Evenimente Speciale")
    evenimente_agregate = df.groupby("eveniment_special")["fare_amount"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x=evenimente_agregate["eveniment_special"], y=evenimente_agregate["fare_amount"], ax=ax)
    ax.set_xlabel("Eveniment special")
    ax.set_ylabel("Tarif mediu")
    st.pyplot(fig)
    st.markdown(
        "**Interpretare:** Evenimentele precum Revelion sau Halloween au tarife medii mai mari datorita cererii crescute si a tarifelor dinamice.")

if aplica_codificare_ciclică:
    st.subheader("Analiza Ciclică a Tarifelor")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ore = np.arange(0, 24)
    zile = np.arange(0, 7)
    medii_ore = df.groupby('ora')['fare_amount'].mean().reset_index()
    medii_zile = df.groupby('zi_saptamana')['fare_amount'].mean().reset_index()

    sns.lineplot(x=ore, y=np.sin(2 * np.pi * ore / 24), ax=ax1, color='blue', label='Sin')
    sns.lineplot(x=ore, y=np.cos(2 * np.pi * ore / 24), ax=ax1, color='red', label='Cos')
    ax2x = ax1.twinx()
    sns.lineplot(x=medii_ore['ora'], y=medii_ore['fare_amount'], ax=ax2x, color='green', label='Tarif Mediu')
    ax1.set_xlabel('Ora')
    ax1.set_ylabel('Valoare Codificare')
    ax2x.set_ylabel('Tarif Mediu ($)')
    ax1.set_title('Codificarea Ciclică și Tarife Medii pe Oră')

    sns.lineplot(x=zile, y=np.sin(2 * np.pi * zile / 7), ax=ax2, color='blue', label='Sin')
    sns.lineplot(x=zile, y=np.cos(2 * np.pi * zile / 7), ax=ax2, color='red', label='Cos')
    ax2y = ax2.twinx()
    sns.lineplot(x=medii_zile['zi_saptamana'], y=medii_zile['fare_amount'], ax=ax2y, color='green', label='Tarif Mediu')
    ax2.set_xlabel('Ziua Săptămânii')
    ax2.set_ylabel('Valoare Codificare')
    ax2y.set_ylabel('Tarif Mediu ($)')
    ax2.set_title('Codificarea Ciclică și Tarife Medii pe Zi')

    fig.legend(loc='upper right')
    st.pyplot(fig)

st.subheader("Harti Locatii Populare")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

gdf_pickup = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.pickup_longitude, df.pickup_latitude),
    crs='EPSG:4326'
).to_crs(epsg=3857)

gdf_pickup.plot(ax=ax1, markersize=0.1, alpha=0.5, color='blue')
ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
ax1.set_title('Locatii Populare de Ridicare')
ax1.set_axis_off()

gdf_dropoff = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.dropoff_longitude, df.dropoff_latitude),
    crs='EPSG:4326'
).to_crs(epsg=3857)

gdf_dropoff.plot(ax=ax2, markersize=0.1, alpha=0.5, color='red')
ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
ax2.set_title('Locatii Populare de Lasare')
ax2.set_axis_off()

st.pyplot(fig)
st.markdown(
    "**Interpretare Ridicari:** Concentratia ridicarilor este mai mare in zone centrale precum aeroporturi si zone comerciale. Activitatea este intensa in zona Manhattan.")
st.markdown(
    "**Interpretare Lasari:** Zonele de lasare coincid partial cu cele de ridicare, cu distributie similara, sugerand cerere concentrata in aceleasi zone urbane.")


# ====================
# 1) CLUSTERIZARE GEOGRAFICĂ (pickup-hotspots)
# ====================
# după calculul coloanei 'distanta_km', imediat înainte de secțiunile de vizualizare:

# 1.a. Convertim coordonatele într-un array (lat/lon în radiani)
coords = df[['pickup_latitude', 'pickup_longitude']].to_numpy()
coords_rad = np.radians(coords)

# 1.b. Definim DBSCAN cu distanță maximă de 0.5 km și min_samples=50
kms_per_radian = 6371.0088
epsilon = 0.5 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=50, algorithm='ball_tree', metric='haversine')
labels = db.fit_predict(coords_rad)

df['pickup_cluster'] = labels
num_clusters = len(set(labels) - {-1})
st.markdown(f"**Hotspots pickup identificate:** {num_clusters} clustere.")

# 1.c. Vizualizare pe hartă
if st.checkbox("Arată pickup-hotspots pe hartă", value=False):
    # filtrăm doar punctele care nu fac parte din zgomot (label != -1)
    df_pick_clusters = df[df['pickup_cluster'] != -1].copy()
    gdf_pick = gpd.GeoDataFrame(
        df_pick_clusters,
        geometry=gpd.points_from_xy(
            df_pick_clusters['pickup_longitude'],
            df_pick_clusters['pickup_latitude']
        ),
        crs='EPSG:4326'
    ).to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf_pick.plot(
        ax=ax,
        column='pickup_cluster',
        categorical=True,
        legend=True,
        markersize=5,
        alpha=0.6
    )
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    st.pyplot(fig)

# ====================
# 2) REGRESIE: PREDICȚIA TARIFULUI
# ====================
# după secțiunea de încărcare df și înainte de afișarea preview-ului:

# 2.a. Pregătire set de date pentru regresie
# selectăm doar rândurile fără valori extreme marcate drept outlier_tarif
df_reg = df[~df['outlier_tarif']].copy()

# definim feature-urile
features = ['distanta_km', 'ora_sin', 'ora_cos', 'zi_sin', 'zi_cos', 'passenger_count']
if 'eveniment_special' in df_reg.columns:
    # codificăm categorical
    df_reg = pd.get_dummies(df_reg, columns=['eveniment_special'], drop_first=True)
    features += [c for c in df_reg.columns if c.startswith('eveniment_special_')]

X = df_reg[features]
y = df_reg['fare_amount']

# 2.b. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2.c. Antrenare model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 2.d. Evaluare model
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.markdown(f"**Regresie RF:** RMSE pe test set = {rmse:.3f}")

# 2.e. Interfață Streamlit pentru predicții
st.sidebar.markdown("### Predictie Tarif")
input_dist = st.sidebar.number_input("Distanță (km)", min_value=0.0, value=1.0)
input_hour = st.sidebar.slider("Ora zilei", 0, 23, 12)
input_pass = st.sidebar.number_input("Număr pasageri", min_value=1, max_value=6, value=1)

# calcul cyclic
sin_h = np.sin(2 * np.pi * input_hour / 24)
cos_h = np.cos(2 * np.pi * input_hour / 24)
# ziua saptamanii asumata neutra (0=Luni)
sin_d = 0.0
cos_d = 1.0

# construire vector de input
user_feat = {
    'distanta_km': input_dist,
    'ora_sin': sin_h,
    'ora_cos': cos_h,
    'zi_sin': sin_d,
    'zi_cos': cos_d,
    'passenger_count': input_pass
}
# completam eveniment_special_dummies cu 0 daca este cazul
for f in features:
    if f not in user_feat:
        user_feat[f] = 0.0

df_user = pd.DataFrame([user_feat])

if st.sidebar.button("Calculează tarif estimat"):
    pred = rf.predict(df_user[features])[0]
    st.sidebar.success(f"Tarif estimat: {pred:.2f}")
st.markdown(
    "Aplicația utilizează Streamlit pentru afișare și analize diverse, împreună cu pandas pentru prelucrarea datelor și Matplotlib/Seaborn pentru reprezentările grafice.")