import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import math

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

st.subheader("Tendințe în Funcție de Timp")
fig, ax = plt.subplots()
sns.boxplot(x=df["categorie_timp"], y=df["fare_amount"], ax=ax)
ax.set_xlabel("Categorie de timp")
ax.set_ylabel("Tarif")
st.pyplot(fig)

st.subheader("Relația dintre Numărul de Pasageri și Tarif")
fig, ax = plt.subplots()
sns.boxplot(x=df["passenger_count"], y=df["fare_amount"], ax=ax)
ax.set_xlabel("Număr de pasageri")
ax.set_ylabel("Tarif")
st.pyplot(fig)

st.subheader("Relația dintre Distanță și Tarif")
fig, ax = plt.subplots()
sns.scatterplot(x=df["distanta_km"], y=df["fare_amount"], hue=df["outlier_tarif"], palette="coolwarm", ax=ax)
ax.set_xlabel("Distanță (km)")
ax.set_ylabel("Tarif")
st.pyplot(fig)

st.subheader("Tarif Mediu pe Zi a Săptămânii")
fig, ax = plt.subplots()
sns.barplot(x=agregare_zi["zi_saptamana"], y=agregare_zi["fare_amount"], ax=ax)
ax.set_xlabel("Zi a săptămânii (0 = Luni, 6 = Duminică)")
ax.set_ylabel("Tarif mediu")
st.pyplot(fig)

st.subheader("Distanța Medie pe An")
fig, ax = plt.subplots()
sns.lineplot(x=agregare_an["an"], y=agregare_an["distanta_km"], marker='o', ax=ax)
ax.set_xlabel("An")
ax.set_ylabel("Distanță medie (km)")
st.pyplot(fig)

st.subheader("Tarif Mediu în Funcție de Numărul de Pasageri")
fig, ax = plt.subplots()
sns.barplot(x=agregare_pasageri["passenger_count"], y=agregare_pasageri["fare_amount"], ax=ax)
ax.set_xlabel("Număr de pasageri")
ax.set_ylabel("Tarif mediu")
st.pyplot(fig)

if aplica_indicator_evenimente:
    st.subheader("Tarif Mediu în Funcție de Evenimente Speciale")
    evenimente_agregate = df.groupby("eveniment_special")["fare_amount"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x=evenimente_agregate["eveniment_special"], y=evenimente_agregate["fare_amount"], ax=ax)
    ax.set_xlabel("Eveniment special")
    ax.set_ylabel("Tarif mediu")
    st.pyplot(fig)

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

st.markdown(
    "Aplicația utilizează Streamlit pentru afișare și analize diverse, împreună cu pandas pentru prelucrarea datelor și Matplotlib/Seaborn pentru reprezentările grafice.")