#PARTE 1: ASYNCIO - Lectura Concurrente de APIs REST (ThingSpeak)
#Tecnología: asyncio + aiohttp
#Función: Descarga de forma asíncrona y simultánea las lecturas de dos estaciones
#         meteorológicas públicas distintas en la región de Boston, Massachusetts
#         (Natick, MA y West Newton, MA) durante el mismo periodo de lluvias
#         (del 12 al 14 de Octubre de 2022).
#Salida: Archivo único 'datos.json'

import asyncio
import aiohttp
import json
import time
import os

# Dos estaciones meteorológicas distintas en el Área Metropolitana de Boston, MA (a 11 km entre sí y ~20 km del Aeropuerto BOS)
# 1. Canal 12397: Estación Meteorológica en Natick, MA (Lat: 42.30, Lon: -71.35)
# 2. Canal 1109290: Estación Meteorológica en West Newton, MA (Lat: 42.36, Lon: -71.22)
CANALES = [
    {
        "id": 12397,
        "nombre": "Estacion_Natick_Boston_MA",
        "ubicacion": "Natick, MA (Metro Boston)",
        "campo_temp": "field4",
        "campo_hum": "field3",
        "campo_pres": "field6",
        "campo_viento": "field2",
        "campo_lluvia": "field5",
        "campo_luz": "field8",
        "params_extra": "&end=2022-10-14%2020:00:00"
    },
    {
        "id": 1109290,
        "nombre": "Estacion_West_Newton_Boston_MA",
        "ubicacion": "West Newton, MA (Metro Boston)",
        "campo_temp": "field1",
        "campo_hum": "field2",
        "campo_pres": "field4",
        "campo_viento": None,
        "campo_lluvia": None,
        "campo_luz": "field3",
        "params_extra": "&end=2022-10-14%2020:00:00"
    }
]

# Cantidad de lecturas a descargar por cada canal
NUM_RESULTADOS = 2000


# Corrutina que descarga los datos de un canal de forma asíncrona
async def descargar_canal(session, canal):
    extra = canal.get("params_extra", "")
    # Armamos la URL del endpoint de ThingSpeak
    url = f"https://api.thingspeak.com/channels/{canal['id']}/feeds.json?results={NUM_RESULTADOS}{extra}"
    print(f"  [Asyncio] Descargando: {canal['nombre']} (ID: {canal['id']})...")
    try:
        # Petición HTTP GET no bloqueante con límite de 15 segundos
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                # Esperamos a que se convierta la respuesta a JSON
                data = await resp.json()
                # Extraemos solo la lista de lecturas (feeds)
                feeds = data.get("feeds", [])
                print(f" Recibidas {len(feeds)} lecturas de {canal['nombre']}")
                # Regresamos los metadatos del canal junto con sus lecturas
                return {
                    "canal_id": canal["id"],
                    "nombre": canal["nombre"],
                    "ubicacion": canal["ubicacion"],
                    "campo_temp": canal["campo_temp"],
                    "campo_hum": canal["campo_hum"],
                    "campo_pres": canal["campo_pres"],
                    "campo_viento": canal.get("campo_viento"),
                    "campo_lluvia": canal.get("campo_lluvia"),
                    "campo_luz": canal.get("campo_luz"),
                    "feeds": feeds
                }
    except Exception as e:
        print(f" Error descargando canal {canal['id']}: {e}")
    return None


# Función principal asíncrona que lanza todas las descargas al mismo tiempo
async def descargar_todos():
    print("Parte 1 del pipeline (async)")
    print("=" * 65)
    # Iniciamos el cronómetro para medir cuánto tarda la descarga concurrente
    t_inicio = time.perf_counter()

    # Abrimos una sola sesión HTTP para reutilizar la conexión
    async with aiohttp.ClientSession() as session:
        # Creamos la lista de tareas pendientes (una por cada estación distinta de la región)
        tareas = [descargar_canal(session, canal) for canal in CANALES]
        # Ejecutamos todas las tareas simultáneamente en paralelo con gather
        resultados = await asyncio.gather(*tareas)

    # Filtramos por si algún canal falló y devolvió None
    resultados_validos = [r for r in resultados if r is not None]

    # Guardamos los datos crudos descargados en datos.json
    ruta_salida = os.path.join(os.path.dirname(__file__), "datos.json")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultados_validos, f, indent=2)

    # Detenemos el cronómetro y contamos el total de lecturas
    t_fin = time.perf_counter()
    total_lecturas = sum(len(r["feeds"]) for r in resultados_validos)
    
    print("-" * 65)
    print(f"  [Asyncio] Completado en: {t_fin - t_inicio:.2f} s")
    print(f"  [Asyncio] Total lecturas: {total_lecturas:,}")
    print(f"  [Asyncio] Guardado en: datos.json")
    print("=" * 65 + "\n")
    return resultados_validos


if __name__ == "__main__":
    # Arrancamos el bucle de eventos de asyncio
    asyncio.run(descargar_todos())
