#PARTE 5: DASHBOARD WEB INTERACTIVO (STREAMLIT)
#Tecnología: streamlit + matplotlib
#Comando de ejecución: streamlit run app_web.py
#Función: Panel visual que consume 'datos.json' y 'vuelos_airlabs.json'.

import streamlit as st
import pandas as pd
import os
import sys
import subprocess
import matplotlib.pyplot as plt

# Configuración general de la página web en Streamlit (título y diseño ancho)
st.set_page_config(
    page_title="Monitoreo Ambiental y Trafico Aereo",
    layout="wide"
)

# Rutas hacia los archivos JSON generados por el pipeline
DIR_ACTUAL = os.path.dirname(__file__)
RUTA_JSON = os.path.join(DIR_ACTUAL, "datos.json")
RUTA_VUELOS = os.path.join(DIR_ACTUAL, "vuelos_airlabs.json")

# Barra lateral izquierda con controles para configurar MPI y ejecutar el pipeline completo
st.sidebar.title("Panel de Control")
num_procesos_mpi = st.sidebar.slider("Procesos MPI:", min_value=1, max_value=8, value=4)

# Si el usuario presiona el botón, corremos las etapas del pipeline en segundo plano
if st.sidebar.button("Ejecutar Pipeline Completo", use_container_width=True):
    with st.spinner("Ejecutando pipeline completo..."):
        mpiexec_path = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
        cmd_mpi = mpiexec_path if os.path.exists(mpiexec_path) else "mpiexec"
        # Ejecutamos en orden: Parte 1 (Asyncio) -> Parte 2 (Pandas) -> Parte 3 (MPI4Py) -> Parte 4 (Numba) -> AirLabs
        subprocess.run([sys.executable, os.path.join(DIR_ACTUAL, "parte1_asyncio.py")], check=True)
        subprocess.run([sys.executable, os.path.join(DIR_ACTUAL, "parte2_pandas.py")], check=True)
        subprocess.run([cmd_mpi, "-n", str(num_procesos_mpi), sys.executable, os.path.join(DIR_ACTUAL, "parte3_mpi.py")], check=True)
        subprocess.run([sys.executable, os.path.join(DIR_ACTUAL, "parte4_numba.py")], check=True)
        subprocess.run([sys.executable, os.path.join(DIR_ACTUAL, "parte_airlabs.py")], check=True)
        st.sidebar.success("Pipeline ejecutado con exito.")

# Encabezado principal del dashboard
st.title("Analisis de Sensores Ambientales (ThingSpeak) y Vuelos (AirLabs)")
st.caption("Computo Paralelo | Asyncio - Pandas - Numba - MPI4Py | Fuente: datos.json y vuelos_airlabs.json")

# Validamos que exista datos.json antes de intentar graficar
if not os.path.exists(RUTA_JSON):
    st.warning("No se encontro 'datos.json'. Haz clic en 'Ejecutar Pipeline Completo'.")
    st.stop()

# Cargamos datos.json en un DataFrame de Pandas
df = pd.read_json(RUTA_JSON)
if "fecha_hora" not in df.columns or "temperatura_c" not in df.columns:
    st.info("El archivo 'datos.json' contiene datos crudos. Ejecuta el pipeline para estructurarlo.")
    st.stop()

# Convertimos la columna fecha_hora a tipo datetime para poder graficar series de tiempo
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

# Creamos 4 columnas para mostrar las tarjetas de indicadores (KPIs)
c1, c2, c3, c4 = st.columns(4)
total_reg = len(df)
temp_med = df["temperatura_c"].mean()
hum_med = df["humedad"].mean()
anomalias = int(df["es_anomalia"].sum()) if "es_anomalia" in df.columns else 0

c1.metric("Lecturas", f"{total_reg:,}")
c2.metric("Temp. Promedio", f"{temp_med:.1f} °C")
c3.metric("Humedad Promedio", f"{hum_med:.1f} %")
c4.metric("Anomalias (Z-Score)", f"{anomalias}")

if anomalias > 0:
    st.warning(f"Se detectaron {anomalias} lecturas anomalas mediante el kernel Numba JIT.")

st.markdown("---")

# Selector desplegable para filtrar por estación meteorológica
estaciones = df["estacion"].unique()
est_sel = st.selectbox("Selecciona una estacion ambiental:", estaciones)
df_est = df[df["estacion"] == est_sel].sort_values("fecha_hora").reset_index(drop=True)

# Gráfica con Matplotlib: Temperatura real, Sensación térmica (Numba) y Anomalías
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_est["fecha_hora"], df_est["temperatura_c"], label="Temperatura (°C)", color="#1f77b4", linewidth=1.5)

if "sensacion_termica_c" in df_est.columns:
    ax.plot(df_est["fecha_hora"], df_est["sensacion_termica_c"], label="Sensacion Termica (Numba)", color="#ff7f0e", linestyle="--", alpha=0.7)

if "es_anomalia" in df_est.columns:
    anom_df = df_est[df_est["es_anomalia"] == 1]
    if not anom_df.empty:
        ax.scatter(anom_df["fecha_hora"], anom_df["temperatura_c"], color="red", s=40, label="Anomalia", zorder=5)

ax.set_ylabel("Temperatura (°C)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")
plt.xticks(rotation=20)
st.pyplot(fig)

# Tabla de lluvia y número de vuelos retrasados en ese mismo periodo de tiempo (AirLabs)
st.markdown("---")
st.subheader("Periodos de Lluvia y Numero de Vuelos Retrasados (AirLabs)")

if os.path.exists(RUTA_VUELOS):
    df_v = pd.read_json(RUTA_VUELOS)
    # Filtramos los momentos en los que llovió y agrupamos por hora/intervalo para mostrar tiempo de lluvia y vuelos retrasados
    df_lluvia = df_v[df_v["lluvia_in"] > 0].copy()
    df_lluvia["periodo_lluvia"] = pd.to_datetime(df_lluvia["fecha_hora"]).dt.strftime("%Y-%m-%d %H:00")

    tabla_lluvia = df_lluvia.groupby("periodo_lluvia").agg(
        tiempo_inicio_lluvia=("fecha_hora", "min"),
        tiempo_fin_lluvia=("fecha_hora", "max"),
        lecturas_con_lluvia=("lluvia_in", "count"),
        lluvia_acumulada_in=("lluvia_in", "sum"),
        numero_vuelos_retrasados=("estado", lambda s: int((s != "A Tiempo").sum())),
        retraso_promedio_min=("retraso_min", "mean")
    ).reset_index()

    tabla_lluvia["retraso_promedio_min"] = tabla_lluvia["retraso_promedio_min"].round(1)
    tabla_lluvia["lluvia_acumulada_in"] = tabla_lluvia["lluvia_acumulada_in"].round(3)

    # Gráfica comparativa Lluvia vs Vuelos Retrasados para confirmar visualmente la relación
    df_v["periodo_hora"] = pd.to_datetime(df_v["fecha_hora"]).dt.floor("h")
    resumen_horas = df_v.groupby("periodo_hora").agg(
        lluvia_total=("lluvia_in", "sum"),
        vuelos_retrasados=("estado", lambda s: int((s != "A Tiempo").sum())),
        retraso_promedio=("retraso_min", "mean")
    ).reset_index()

    fig_lluvia, ax_lluvia = plt.subplots(figsize=(12, 4))
    ax_vuelos = ax_lluvia.twinx()

    # Barras azules para la lluvia registrada por el sensor y línea roja para el retraso de los vuelos
    ax_lluvia.bar(resumen_horas["periodo_hora"], resumen_horas["lluvia_total"], width=0.03, color="#1f77b4", alpha=0.5, label="Lluvia Registrada (in)")
    ax_vuelos.plot(resumen_horas["periodo_hora"], resumen_horas["retraso_promedio"], color="#d62728", linewidth=2, marker="o", markersize=4, label="Retraso Promedio Vuelos (min)")

    ax_lluvia.set_ylabel("Lluvia Acumulada por Hora (in)", color="#1f77b4")
    ax_vuelos.set_ylabel("Retraso Promedio de Vuelos (min)", color="#d62728")
    ax_lluvia.set_title("Comparativa Temporal: Lluvia en el Sensor vs Retraso de Vuelos en el Area (Boston)")
    ax_lluvia.grid(True, linestyle=":", alpha=0.6)

    # Unimos las leyendas de ambos ejes en una sola caja
    lineas1, etiquetas1 = ax_lluvia.get_legend_handles_labels()
    lineas2, etiquetas2 = ax_vuelos.get_legend_handles_labels()
    ax_vuelos.legend(lineas1 + lineas2, etiquetas1 + etiquetas2, loc="upper left")
    plt.xticks(rotation=20)
    st.pyplot(fig_lluvia)

    st.dataframe(tabla_lluvia, use_container_width=True)
else:
    st.info("Ejecuta parte_airlabs.py o el pipeline completo para generar la tabla de vuelos retrasados por lluvia.")

# Tabla de datos de los sensores
st.markdown("---")
st.subheader("Registros Procesados")
cols_mostrar = [c for c in ["fecha_hora", "estacion", "temperatura_c", "sensacion_termica_c", "punto_rocio_c", "humedad", "lluvia_in", "es_anomalia", "procesado_por_rank"] if c in df.columns]
st.dataframe(df[cols_mostrar].head(100), use_container_width=True)
