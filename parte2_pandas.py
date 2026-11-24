#PARTE 2: PANDAS - Limpieza y Transformación Tabular
#Tecnología: pandas
#Función: Lee 'datos.json', normaliza formatos, convierte temperaturas a Celsius,
#         limpia valores atípicos y nulos, e interpola datos faltantes.
#Salida: Actualiza 'datos.json' con la tabla limpia y estructurada.

import json
import os
import pandas as pd
import numpy as np


def procesar_datos():
    print("Parte 2 del pipeline (pandas)")
    print("=" * 65)
    
    dir_actual = os.path.dirname(__file__)
    ruta_archivo = os.path.join(dir_actual, "datos.json")
    
    # Verificamos que exista el archivo generado por la parte 1
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError("No se encontró 'datos.json'. Ejecuta primero parte1_asyncio.py")
        
    # Cargamos el JSON con los datos crudos de los canales
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        canales_data = json.load(f)
        
    registros_limpios = []
    
    # Recorremos cada canal descargado para limpiar y unificar sus columnas
    for canal in canales_data:
        cid = canal["canal_id"]
        nombre = canal["nombre"]
        # Leemos qué field corresponde a cada variable en este canal
        campo_temp = canal["campo_temp"]
        campo_hum = canal["campo_hum"]
        campo_pres = canal["campo_pres"]
        campo_viento = canal.get("campo_viento")
        campo_lluvia = canal.get("campo_lluvia")
        campo_luz = canal.get("campo_luz")
        feeds = canal["feeds"]
        
        # Convertimos la lista de lecturas del canal en un DataFrame de Pandas
        df_canal = pd.DataFrame(feeds)
        if df_canal.empty:
            continue
            
        # 1. Fechas y metadatos: convertimos el texto created_at a formato fecha
        df_canal["fecha_hora"] = pd.to_datetime(df_canal["created_at"], errors="coerce")
        df_canal["canal_id"] = cid
        df_canal["estacion"] = nombre
        
        # 2. Conversión a numérico: si hay texto inválido lo convierte en NaN (coerce)
        df_canal["temperatura"] = pd.to_numeric(df_canal[campo_temp], errors="coerce") if campo_temp in df_canal.columns else np.nan
        df_canal["humedad"] = pd.to_numeric(df_canal[campo_hum], errors="coerce") if campo_hum in df_canal.columns else np.nan
        df_canal["presion"] = pd.to_numeric(df_canal[campo_pres], errors="coerce") if (campo_pres and campo_pres in df_canal.columns) else np.nan
        df_canal["viento_mph"] = pd.to_numeric(df_canal[campo_viento], errors="coerce") if (campo_viento and campo_viento in df_canal.columns) else 0.0
        df_canal["lluvia_in"] = pd.to_numeric(df_canal[campo_lluvia], errors="coerce") if (campo_lluvia and campo_lluvia in df_canal.columns) else 0.0
        df_canal["nivel_luz"] = pd.to_numeric(df_canal[campo_luz], errors="coerce") if (campo_luz and campo_luz in df_canal.columns) else 500.0
        
        # 3. Conversión de unidades: el canal 12397 viene en Fahrenheit, lo pasamos a Celsius
        if cid == 12397:
            df_canal["temperatura_c"] = (df_canal["temperatura"] - 32) * (5.0 / 9.0)
        else:
            df_canal["temperatura_c"] = df_canal["temperatura"]
            
        # 4. Filtro de rangos válidos: descartamos lecturas físicamente imposibles
        df_canal.loc[(df_canal["temperatura_c"] < -40) | (df_canal["temperatura_c"] > 60), "temperatura_c"] = np.nan
        df_canal.loc[(df_canal["humedad"] < 0) | (df_canal["humedad"] > 100), "humedad"] = np.nan
        
        # 5. Interpolación de nulos: ordenamos por fecha y rellenamos huecos linealmente
        df_canal = df_canal.sort_values("fecha_hora").reset_index(drop=True)
        df_canal["temperatura_c"] = df_canal["temperatura_c"].interpolate(method="linear").bfill().ffill()
        df_canal["humedad"] = df_canal["humedad"].interpolate(method="linear").bfill().ffill()
        df_canal["presion"] = df_canal["presion"].interpolate(method="linear").bfill().ffill().fillna(1013.25)
        df_canal["viento_mph"] = df_canal["viento_mph"].fillna(0.0)
        df_canal["lluvia_in"] = df_canal["lluvia_in"].fillna(0.0)
        df_canal["nivel_luz"] = df_canal["nivel_luz"].fillna(500.0)
        
        # Seleccionamos únicamente las columnas limpias y estandarizadas
        cols = ["fecha_hora", "canal_id", "estacion", "temperatura_c", "humedad", "presion", "viento_mph", "lluvia_in", "nivel_luz"]
        registros_limpios.append(df_canal[cols])
        
    # Unimos los DataFrames de todos los canales en una sola tabla general
    df_completo = pd.concat(registros_limpios, ignore_index=True)
    df_completo = df_completo.sort_values("fecha_hora").reset_index(drop=True)
    
    # Guardamos la tabla limpia sobrescribiendo datos.json
    df_completo["fecha_hora"] = df_completo["fecha_hora"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df_completo.to_json(ruta_archivo, orient="records", indent=2)
    
    print(f"  [Pandas] Registros limpios: {len(df_completo):,}")
    print(f"  [Pandas] Rango temp: {df_completo['temperatura_c'].min():.1f} °C a {df_completo['temperatura_c'].max():.1f} °C")
    print(f"  [Pandas] Humedad promedio: {df_completo['humedad'].mean():.1f} %")
    print(f"  [Pandas] Guardado en: datos.json")
    print("=" * 65 + "\n")
    return df_completo


if __name__ == "__main__":
    procesar_datos()
