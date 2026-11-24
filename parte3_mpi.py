#PARTE 3: MPI4PY - Distribución del Procesamiento en Múltiples Procesos
#Tecnología: mpi4py
#Comando de ejecución: mpiexec -n 4 python parte3_mpi.py
#Función:
#  1. Rank 0 lee 'datos.json' limpio de la Parte 2 (Pandas) y divide los datos en 4 partes iguales.
#  2. Distribuye un bloque a cada proceso con 'comm.scatter()'.
#  3. Cada proceso procesa su bloque asignado y le marca su número de proceso ('procesado_por_rank').
#  4. Rank 0 recolecta los bloques con 'comm.gather()' y consolida en 'datos.json' para la Parte 4 (Numba).
#Salida: Actualiza 'datos.json' con los datos distribuidos por proceso.

import time
import os
import numpy as np
import pandas as pd
from mpi4py import MPI


def ejecutar_mpi():
    # Inicializamos el comunicador de MPI, obtenemos el ID del proceso (rank) y el total de procesos (size)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    dir_actual = os.path.dirname(__file__)
    ruta_archivo = os.path.join(dir_actual, "datos.json")
    
    chunks = None
    t_inicio_global = 0.0
    
    # El proceso maestro (Rank 0) lee datos.json de Pandas y lo divide en partes iguales (chunks)
    if rank == 0:
        print("Parte 3 del pipeline (mpi4py)")
        print("=" * 65)
        print(f"  [MPI] Numero de procesos : {size}")
        
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError("No se encontro 'datos.json'. Ejecuta primero parte2_pandas.py")
            
        df = pd.read_json(ruta_archivo)
        total_filas = len(df)
        print(f"  [MPI] Registros a distribuir : {total_filas:,}")
        
        # Dividimos las filas equitativamente entre el número de procesos (4000 / 4 = 1000 por proceso)
        indices_division = np.array_split(range(total_filas), size)
        chunks = [df.iloc[idx].copy() for idx in indices_division]
        # Iniciamos el cronómetro global en el proceso maestro
        t_inicio_global = time.perf_counter()

    # Scatter: Rank 0 reparte un pedazo (chunk_local) a cada proceso de forma paralela
    chunk_local = comm.scatter(chunks, root=0)
    
    # Iniciamos el cronómetro local de este proceso (rank)
    t_inicio_local = time.perf_counter()
    n_local = len(chunk_local)
    
    # Cada proceso recibe su bloque, verifica sus datos locales y registra qué Rank los procesó
    if n_local > 0:
        chunk_local["procesado_por_rank"] = rank
        temp_media_local = float(chunk_local["temperatura_c"].mean())
    else:
        temp_media_local = 0.0
        
    # Detenemos el cronómetro local del proceso
    t_fin_local = time.perf_counter()
    tiempo_local = t_fin_local - t_inicio_local

    # Resumen de métricas de este proceso para enviarlo al Rank 0
    estadisticas_locales = {
        "rank": rank,
        "registros": n_local,
        "temp_media": temp_media_local,
        "tiempo_ms": tiempo_local * 1000
    }

    # Gather: Rank 0 recolecta todos los pedazos distribuidos y las estadísticas de los demás procesos
    todos_los_chunks = comm.gather(chunk_local, root=0)
    todas_las_estadisticas = comm.gather(estadisticas_locales, root=0)

    # El proceso maestro une todos los pedazos y guarda datos.json listo para la Parte 4 (Numba)
    if rank == 0:
        t_fin_global = time.perf_counter()
        tiempo_global = t_fin_global - t_inicio_global
        
        df_final = pd.concat(todos_los_chunks, ignore_index=True)
        df_final.to_json(ruta_archivo, orient="records", indent=2)
        
        print("-" * 65)
        print(f"{'Proceso (Rank)':<16} | {'Registros':<12} | {'Temp Media':<12} | {'Tiempo (ms)':<12}")
        print("-" * 65)
        for stat in todas_las_estadisticas:
            print(f"Rank {stat['rank']:<11} | {stat['registros']:<12,} | {stat['temp_media']:<12.2f} | {stat['tiempo_ms']:<12.3f}")
        print("-" * 65)
        print(f"  [MPI] Tiempo total: {tiempo_global * 1000:.2f} ms")
        print(f"  [MPI] Guardado consolidado en: datos.json")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    ejecutar_mpi()
