import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import geopandas as gpd
import contextily as ctx
from joblib import Parallel, delayed

from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os

# ========================================
# Secțiunea Sidebar
# ========================================
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


# ========================================
# Încărcare și preprocesare date
# ========================================
@st.cache_data
def incarca_date_brute():
    """Încarcă datele brute cu tipuri optimizate de date"""
    dtypes = {
        'fare_amount': 'float32',
        'pickup_longitude': 'float32',
        'pickup_latitude': 'float32',
        'dropoff_longitude': 'float32',
        'dropoff_latitude': 'float32',
        'passenger_count': 'int8'
    }
    df = pd.read_csv("uber.csv", parse_dates=['pickup_datetime'], dtype=dtypes)

    # Elimină valorile lipsă
    df.dropna(inplace=True)

    # Calculează distanța folosind vectorizare
    coords = df[['pickup_latitude', 'pickup_longitude',
                 'dropoff_latitude', 'dropoff_longitude']].astype('float32').values
    R = 6371
    dlat = np.radians(coords[:, 2] - coords[:, 0])
    dlon = np.radians(coords[:, 3] - coords[:, 1])
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(coords[:, 0])) * np.cos(np.radians(coords[:, 2])) * np.sin(
        dlon / 2) ** 2
    df["distanta_km"] = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Extrage caracteristici temporale
    df["ora"] = df["pickup_datetime"].dt.hour
    df["zi_saptamana"] = df["pickup_datetime"].dt.dayofweek
    df["an"] = df["pickup_datetime"].dt.year

    return df


@st.cache_data
def preproceseaza_date(df, metoda_extreme, aplica_binning_timp, aplica_codificare_ciclică,
                       aplica_indicator_evenimente, metoda_scalare, coloane_scalare):
    """Preprocesează datele cu opțiunile selectate de utilizator"""

    # Funcție pentru calcul IQR (folosită în paralel)
    def calculeaza_iqr(col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    coloane_verificare = ["fare_amount", "passenger_count", "pickup_longitude", "pickup_latitude",
                          "dropoff_longitude", "dropoff_latitude"]

    # Procesare valori extreme în paralel
    if metoda_extreme == "Eliminare IQR":
        rezultate = Parallel(n_jobs=4)(delayed(calculeaza_iqr)(col) for col in coloane_verificare)
        masca_totala = pd.Series(True, index=df.index)

        for col, (lower, upper) in zip(coloane_verificare, rezultate):
            masca_coloana = (df[col] >= lower) & (df[col] <= upper)
            masca_totala &= masca_coloana

        df = df[masca_totala]
        st.markdown(f"**Eliminare IQR:** S-au eliminat {len(masca_totala) - masca_totala.sum()} înregistrări.")

    elif metoda_extreme == "Winsorizare":
        rezultate = Parallel(n_jobs=4)(delayed(calculeaza_iqr)(col) for col in coloane_verificare)
        for col, (lower, upper) in zip(coloane_verificare, rezultate):
            df[col] = df[col].clip(lower, upper)
        st.markdown("**Winsorizare:** S-au limitat valorile extreme.")

    elif metoda_extreme == "Transformare logaritmică":
        df = df[df["fare_amount"] > 0.01]
        df["fare_amount"] = np.log1p(df["fare_amount"])
        st.markdown("**Transformare logaritmică:** S-a aplicat log1p pe fare_amount.")

    # Binning timp
    if aplica_binning_timp:
        bins = [0, 6, 12, 18, 24]
        etichete = ["Noapte", "Dimineață", "Prânz", "Seară"]
        df["categorie_timp"] = pd.cut(df["ora"], bins=bins, right=False, labels=etichete)
    else:
        df["categorie_timp"] = df["ora"]

    # Codificare ciclică
    if aplica_codificare_ciclică:
        df["ora_sin"] = np.sin(2 * np.pi * df["ora"] / 24)
        df["ora_cos"] = np.cos(2 * np.pi * df["ora"] / 24)
        df["zi_sin"] = np.sin(2 * np.pi * df["zi_saptamana"] / 7)
        df["zi_cos"] = np.cos(2 * np.pi * df["zi_saptamana"] / 7)

    # Indicator evenimente speciale
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

    # Scalare caracteristici
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

    # Calculează tarif pe km
    df["tarif_per_km"] = df["fare_amount"] / (df["distanta_km"] + 1e-6)
    media_tarif = df["tarif_per_km"].mean()
    std_tarif = df["tarif_per_km"].std()
    prag = media_tarif + 2 * std_tarif
    df["outlier_tarif"] = df["tarif_per_km"] > prag

    return df


@st.cache_data
def calculeaza_agregari(df, aplica_binning_timp):
    """Calculează agregările pentru vizualizări"""
    if aplica_binning_timp:
        agregare_timp = df.groupby("categorie_timp")["fare_amount"].mean().reset_index()
    else:
        agregare_timp = df.groupby("ora")["fare_amount"].mean().reset_index()

    agregare_zi = df.groupby("zi_saptamana")["fare_amount"].mean().reset_index()
    agregare_an = df.groupby("an")["distanta_km"].mean().reset_index()
    agregare_pasageri = df.groupby("passenger_count")["fare_amount"].mean().reset_index()

    return agregare_timp, agregare_zi, agregare_an, agregare_pasageri


# Încărcare date brute
df_brut = incarca_date_brute()

# Preprocesare date
df = preproceseaza_date(
    df_brut,
    metoda_extreme,
    aplica_binning_timp,
    aplica_codificare_ciclică,
    aplica_indicator_evenimente,
    metoda_scalare,
    coloane_scalare
)

# Calculează agregări
agregare_timp, agregare_zi, agregare_an, agregare_pasageri = calculeaza_agregari(df, aplica_binning_timp)

# Eșantion pentru vizualizări grele
eșantion_viz = df.sample(min(10000, len(df)), random_state=42) if len(df) > 10000 else df

# ========================================
# Interfața principală
# ========================================
st.title("Analiza Tarifelor Uber")
st.markdown(
    "Această aplicație prelucrează și analizează datele curselor Uber utilizând Streamlit, pandas și diverse tehnici de prelucrare a datelor.")

with st.expander("Preview Date", expanded=False):
    st.subheader("Preview Date")
    st.write(df.head())

with st.expander("Statistici Descriptive", expanded=False):
    st.subheader("Statistici Descriptive")
    st.write(df.describe())

st.subheader("Distribuția Tarifelor")
fig, ax = plt.subplots()
sns.histplot(df["fare_amount"], bins=50, kde=True, ax=ax)
ax.set_xlabel("Tarif")
ax.set_ylabel("Frecvență")
st.pyplot(fig)
st.markdown(
    "**Interpretare:** Majoritatea curselor au tarife mici intre 4-10 USD. Distributia este asimetrica cu valori extreme peste 12 USD, probabil curse lungi sau la ore de varf.")

st.subheader("Tendințe în Funcție de Timp")
fig, ax = plt.subplots()
sns.boxplot(x=df["categorie_timp"], y=df["fare_amount"], ax=ax)
ax.set_xlabel("Categorie de timp")
ax.set_ylabel("Tarif")
st.pyplot(fig)
st.markdown("**Interpretare:** Distribuția tarifelor este relativ similară între cele patru categorii de timp.")
st.markdown("Noaptea pare să aibă o mediană puțin mai mare și o variabilitate ușor crescută față de celelalte intervale.")
st.markdown("Outlierii sunt frecvenți în toate intervalele, în special tarife foarte mari (peste 20).")
st.markdown("Dimineața și prânzul au mediane aproape identice și o distribuție destul de simetrică.")

st.subheader("Relația dintre Numărul de Pasageri și Tarif")
fig, ax = plt.subplots()
sns.boxplot(x=df["passenger_count"], y=df["fare_amount"], ax=ax)
ax.set_xlabel("Număr de pasageri")
ax.set_ylabel("Tarif")
st.pyplot(fig)
st.markdown(    "**Interpretare:** Nu există un trend clar care să arate că tariful crește odată cu numărul de pasageri.")
st.markdown("Distribuția tarifelor este foarte similară între toate categoriile de număr de pasageri (0–3).")
st.markdown("Mediana tarifelor este aproape identică pentru toate grupurile (~7–8 USD).")
st.markdown("Outlieri există în toate grupurile, indicând curse cu tarife anormal de mici sau mari, dar aceste excepții nu par corelate cu numărul de pasageri.")

st.subheader("Relația dintre Distanță și Tarif")
fig, ax = plt.subplots()
sns.scatterplot(x=eșantion_viz["distanta_km"], y=eșantion_viz["fare_amount"],
                hue=eșantion_viz["outlier_tarif"], palette="coolwarm", alpha=0.5, ax=ax)
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
    "**Interpretare:** Tariful mediu este aproximativ la fel in orice zi, cu o mica crestere in mijlocul saptamanii.")

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
    "**Interpretare:** Cursele cu mai mult de un pasager tind sa fie putin mai scumpe, posibil datorita unor altor factori precum distanta.")

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
    st.markdown("Exista o scadere brusca a numarului de calatorii efectuate intr-o zi, in jurul orei 6 dimineata.")
    st.markdown("Exista cresteri si scaderi bruste in intervalul 8-20, datorita orelor de varf cand lumea merge sau pleaca de la munca.")
    st.markdown("In timpul saptamanii numarul de calatorii realizate este mai mic in timpul weekendului si in prima zi a saptamanii.")
# ========================================
# Hărți cu eșantion redus
# ========================================
with st.expander("Harti Locatii Populare", expanded=False):
    st.subheader("Harti Locatii Populare")
    eșantion_hărți = df.sample(min(60000, len(df)), random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    gdf_pickup = gpd.GeoDataFrame(
        eșantion_hărți,
        geometry=gpd.points_from_xy(eșantion_hărți.pickup_longitude, eșantion_hărți.pickup_latitude),
        crs='EPSG:4326'
    ).to_crs(epsg=3857)

    gdf_pickup.plot(ax=ax1, markersize=0.1, alpha=0.5, color='blue')
    ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
    ax1.set_title('Locatii Populare de Ridicare')
    ax1.set_axis_off()

    gdf_dropoff = gpd.GeoDataFrame(
        eșantion_hărți,
        geometry=gpd.points_from_xy(eșantion_hărți.dropoff_longitude, eșantion_hărți.dropoff_latitude),
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

# ========================================
# Clusterizare geografică
# ========================================
with st.expander("Clusterizare Geografică", expanded=False):
    st.subheader("Clusterizare Geografică")

    # Eșantion pentru clusterizare
    eșantion_cluster = df.sample(min(20000, len(df)), random_state=42)

    # 1.a. Convertim coordonatele
    coords = eșantion_cluster[['pickup_latitude', 'pickup_longitude']].to_numpy()
    coords_rad = np.radians(coords)

    # 1.b. Definim DBSCAN
    kms_per_radian = 6371.0088
    epsilon = 0.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=50, algorithm='ball_tree', metric='haversine')
    labels = db.fit_predict(coords_rad)

    eșantion_cluster['pickup_cluster'] = labels
    num_clusters = len(set(labels) - {-1})
    st.markdown(f"**Hotspots pickup identificate:** {num_clusters} clustere.")

    # 1.c. Vizualizare pe hartă
    if st.checkbox("Arată pickup-hotspots pe hartă", value=False):
        df_pick_clusters = eșantion_cluster[eșantion_cluster['pickup_cluster'] != -1].copy()

        if not df_pick_clusters.empty:
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
        else:
            st.warning("Nu s-au găsit clustere pentru afișare")

# ========================================
# Regresie pentru predicție tarif (versiune îmbunătățită)
# ========================================
with st.expander("Regresie Tarif", expanded=False):
    st.sidebar.markdown("## Regresie Tarif Uber")
    retrain = st.sidebar.button("Reantrenează modelul")
    model_path = "model_ridge.pkl"  # Schimbăm la Ridge pentru stabilitate

    # Pregătesc date pentru regresie
    df_reg = df[~df["outlier_tarif"]].copy()

    # Extrage caracteristici temporale
    df_reg["ora"] = df_reg["pickup_datetime"].dt.hour
    df_reg["zi_saptamana"] = df_reg["pickup_datetime"].dt.dayofweek

    # Crează caracteristici ciclice dacă nu sunt deja create
    if aplica_codificare_ciclică:
        # Folosim direct coloanele existente dacă sunt disponibile
        if "ora_sin" not in df_reg.columns:
            df_reg["ora_sin"] = np.sin(2 * np.pi * df_reg["ora"] / 24)
            df_reg["ora_cos"] = np.cos(2 * np.pi * df_reg["ora"] / 24)
        if "zi_sin" not in df_reg.columns:
            df_reg["zi_sin"] = np.sin(2 * np.pi * df_reg["zi_saptamana"] / 7)
            df_reg["zi_cos"] = np.cos(2 * np.pi * df_reg["zi_saptamana"] / 7)
    else:
        # Crează caracteristici ciclice doar pentru regresie
        df_reg["ora_sin"] = np.sin(2 * np.pi * df_reg["ora"] / 24)
        df_reg["ora_cos"] = np.cos(2 * np.pi * df_reg["ora"] / 24)
        df_reg["zi_sin"] = np.sin(2 * np.pi * df_reg["zi_saptamana"] / 7)
        df_reg["zi_cos"] = np.cos(2 * np.pi * df_reg["zi_saptamana"] / 7)

    # Caracteristici de bază
    features = ["distanta_km", "passenger_count", "ora_sin", "ora_cos", "zi_sin", "zi_cos"]

    # One-hot encoding pentru evenimente speciale
    if aplica_indicator_evenimente and "eveniment_special" in df_reg.columns:
        df_reg = pd.get_dummies(df_reg, columns=["eveniment_special"], drop_first=True)
        ev_cols = [c for c in df_reg.columns if c.startswith("eveniment_special_")]
        features += ev_cols

    # Eșantion pentru antrenament
    eșantion_regresie = df_reg.sample(min(50000, len(df_reg)), random_state=42)
    X = eșantion_regresie[features]
    y = eșantion_regresie["fare_amount"]

    # Scalare caracteristici
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Antrenare sau încărcare model
    if os.path.exists(model_path) and not retrain:
        with open(model_path, "rb") as f:
            model, scaler_state = pickle.load(f)
            # Reconstruim scalerul cu starea salvată
            scaler = StandardScaler()
            scaler.mean_ = scaler_state['mean']
            scaler.scale_ = scaler_state['scale']
        st.sidebar.success("Model Ridge încărcat din disk.")
    else:
        from sklearn.linear_model import Ridge  # Schimbăm la Ridge pentru stabilitate

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Folosim Ridge Regression pentru stabilitate numerică
        model = Ridge(alpha=1.0)  # Model liniar stabil
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.sidebar.write(f"RMSE pe set de test: {rmse:.3f}")

        # Salvăm modelul și starea scalerului
        scaler_state = {
            'mean': scaler.mean_,
            'scale': scaler.scale_
        }
        with open(model_path, "wb") as f:
            pickle.dump((model, scaler_state), f)
        st.sidebar.success(f"Model antrenat și salvat în '{model_path}'")

    # Interfață predicție
    st.sidebar.markdown("### Predicție Tarif")

    # Folosim coloane pentru a organiza mai bine input-urile
    col1, col2 = st.sidebar.columns(2)
    with col1:
        input_dist = st.number_input("Distanță (km)", min_value=0.1, value=3.5, step=0.1)
        input_pass = st.number_input("Număr pasageri", 1, 6, 2)
    with col2:
        input_hour = st.slider("Ora zilei", 0, 23, 5)
        input_day = st.slider("Ziua săptămânii (0=Luni)", 0, 6, 4)  # 4 = Vineri

    # Codificare ciclică pentru input
    sin_h = np.sin(2 * np.pi * input_hour / 24)
    cos_h = np.cos(2 * np.pi * input_hour / 24)
    sin_d = np.sin(2 * np.pi * input_day / 7)
    cos_d = np.cos(2 * np.pi * input_day / 7)

    # Construire date de intrare
    date_intrare = {
        "distanta_km": input_dist,
        "passenger_count": input_pass,
        "ora_sin": sin_h,
        "ora_cos": cos_h,
        "zi_sin": sin_d,
        "zi_cos": cos_d
    }

    # Adăugăm coloane pentru evenimente speciale dacă sunt necesare
    if aplica_indicator_evenimente and "eveniment_special" in df_reg.columns:
        for col in [c for c in df_reg.columns if c.startswith("eveniment_special_")]:
            date_intrare[col] = 0

    # Cream DataFrame-ul pentru predicție
    df_user = pd.DataFrame([date_intrare])
    df_user = df_user.reindex(columns=features, fill_value=0.0)

    # Scalăm datele de intrare
    X_user = scaler.transform(df_user)

    if st.sidebar.button("Calculează tarif estimat", key="predict_button"):
        pred = model.predict(X_user)[0]
        # Verificăm dacă am aplicat transformare logaritmică
        if metoda_extreme == "Transformare logaritmică":
            pred = np.expm1(pred)  # Transformare inversă

        st.sidebar.success(f"Tarif estimat: ${pred:.2f} USD")

st.markdown("---")
st.markdown(
    "Aplicația utilizează Streamlit pentru afișare și analize diverse, împreună cu pandas pentru prelucrarea datelor și Matplotlib/Seaborn pentru reprezentările grafice.")