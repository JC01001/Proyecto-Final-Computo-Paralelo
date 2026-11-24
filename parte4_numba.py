#PARTE 4: NUMBA - Aceleración de Cálculos Numéricos (CPU Multihilo)
#Tecnología: numba (@njit con parallel=True)
#Función: Lee 'datos.json' ya distribuido por la Parte 3 (MPI4Py) y realiza las
#         operaciones numéricas intensivas compiladas a código máquina:
#         - Sensación térmica (Heat Index)
#         - Punto de rocío (Magnus-Tetens)
#         - Detección de anomalías por Z-Score
#Salida: Actualiza 'datos.json' con los resultados calculados.

import time
import math
import numpy as np
import pandas as pd
import os
from numba import njit, prange


# Decorador de Numba que compila la función a código máquina y paraleliza los ciclos prange
@njit(parallel=True, fastmath=True)
def calcular_metricas_ambientales_numba(temps, hums, umbral_z=2.0):
    n = temps.shape[0]
    
    # Calculamos el promedio de temperatura en paralelo
    suma = 0.0
    for i in prange(n):
        suma += temps[i]
    media = suma / n
    
    # Calculamos la desviación estándar en paralelo para el Z-Score
    suma_sq = 0.0
    for i in prange(n):
        diff = temps[i] - media
        suma_sq += diff * diff
    desv = math.sqrt(suma_sq / n)
    if desv == 0.0:
        desv = 1.0
        
    # Reservamos los arreglos de salida en memoria
    sensacion_termica = np.empty(n, dtype=np.float64)
    punto_rocio = np.empty(n, dtype=np.float64)
    anomalias = np.zeros(n, dtype=np.int32)
    
    # Constantes de la fórmula meteorológica de Magnus-Tetens
    a, b = 17.27, 237.7
    
    # Ciclo paralelo principal repartido entre todos los hilos del CPU
    for i in prange(n):
        t = temps[i]
        h = hums[i]
        
        # 1. Sensación térmica: combina temperatura y presión de vapor por humedad
        e_sat = (h / 100.0) * 6.105 * math.exp((a * t) / (b + t))
        sensacion_termica[i] = t + 0.33 * e_sat - 0.7
        
        # 2. Punto de rocío: temperatura a la que se condensa la humedad
        alpha = ((a * t) / (b + t)) + math.log(max(h / 100.0, 0.001))
        punto_rocio[i] = (b * alpha) / (a - alpha)
        
        # 3. Detección de anomalías: si se aleja más de 2 desviaciones estándar (Z-Score)
        z = math.fabs(t - media) / desv
        if z > umbral_z:
            anomalias[i] = 1
            
    return sensacion_termica, punto_rocio, anomalias, media, desv


def ejecutar_analisis_numba():
    print("Parte 4 del pipeline (numba)")
    print("=" * 65)
    
    dir_actual = os.path.dirname(__file__)
    ruta_archivo = os.path.join(dir_actual, "datos.json")
    
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError("No se encontro 'datos.json'. Ejecuta primero parte3_mpi.py")
        
    # Leemos la tabla distribuida de datos.json y extraemos arreglos de NumPy
    df = pd.read_json(ruta_archivo)
    temps = df["temperatura_c"].to_numpy(dtype=np.float64)
    hums = df["humedad"].to_numpy(dtype=np.float64)
    
    # Ejecución del kernel compilado con Numba JIT Parallel
    t0_numba = time.perf_counter()
    sensacion, rocio, anomalias, media_t, desv_t = calcular_metricas_ambientales_numba(temps, hums, umbral_z=2.0)
    t1_numba = time.perf_counter()
    tiempo_numba = t1_numba - t0_numba
    
    # Agregamos las nuevas columnas calculadas al DataFrame
    df["sensacion_termica_c"] = np.round(sensacion, 2)
    df["punto_rocio_c"] = np.round(rocio, 2)
    df["es_anomalia"] = anomalias
    
    total_anomalias = int(np.sum(anomalias))
    
    # Guardamos los resultados enriquecidos en datos.json
    df.to_json(ruta_archivo, orient="records", indent=2)
    
    print("-" * 65)
    print(f"  [Numba] Tiempo de ejecucion  : {tiempo_numba * 1000:.3f} ms")
    print(f"  [Numba] Anomalias detectadas : {total_anomalias}")
    print(f"  [Numba] Guardado en: datos.json")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    ejecutar_analisis_numba()
