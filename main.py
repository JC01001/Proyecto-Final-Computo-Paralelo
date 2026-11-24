#ORQUESTADOR PRINCIPAL DEL PROYECTO
#Ejecuta todo el pipeline secuencialmente desde la terminal:
#1. Parte 1 (Asyncio): Descarga concurrente desde ThingSpeak.
#2. Parte 2 (Pandas): Limpieza y unificación tabular.
#3. Parte 3 (MPI4Py): Distribución paralela en 4 procesos con mpiexec.
#4. Parte 4 (Numba): Aceleración de cálculos numéricos JIT.
#5. Parte AirLabs: Tráfico aéreo sincronizado con el clima (lluvia, visibilidad y retrasos).

import subprocess
import sys
import os
import time


# Función auxiliar que ejecuta cada script del proyecto como un subproceso y mide su tiempo
def ejecutar_script(comando, descripcion):
    print("\n" + "#" * 65)
    print(f" EJECUTANDO: {descripcion}")
    # Iniciamos el cronómetro antes de lanzar el script
    t0 = time.perf_counter()
    # Ejecutamos el comando en la terminal y esperamos a que termine
    res = subprocess.run(comando)
    # Detenemos el cronómetro al terminar el script
    t1 = time.perf_counter()
    # Si el código de salida es distinto de 0, hubo un error y detenemos el pipeline
    if res.returncode != 0:
        print(f"no se ejecuto {descripcion}")
        sys.exit(res.returncode)
    # Imprimimos cuántos segundos tardó en completarse esta etapa
    print(f"{descripcion} finalizado en {t1 - t0:.2f} s")


def main():
    # Obtenemos la ruta de la carpeta actual donde viven todos los scripts
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Parte 1 (Asyncio): descarga concurrente de los sensores de ThingSpeak hacia datos.json
    ejecutar_script([sys.executable, os.path.join(dir_actual, "parte1_asyncio.py")], "PARTE 1: Asyncio (Descarga Concurrente ThingSpeak)")
    
    # 2. Parte 2 (Pandas): limpieza tabular, conversión a Celsius e interpolación sobre datos.json
    ejecutar_script([sys.executable, os.path.join(dir_actual, "parte2_pandas.py")], "PARTE 2: Pandas (Limpieza y Estructuracion)")
    
    # 3. Parte 3 (MPI4Py): distribución del procesamiento en 4 procesos paralelos usando mpiexec
    mpiexec_path = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
    cmd_mpi = mpiexec_path if os.path.exists(mpiexec_path) else "mpiexec"
    ejecutar_script([cmd_mpi, "-n", "4", sys.executable, os.path.join(dir_actual, "parte3_mpi.py")], "PARTE 3: MPI4Py (Distribucion con 4 Procesos)")

    # 4. Parte 4 (Numba): aceleración de cálculos numéricos en CPU (sensación térmica, rocío y anomalías Z-Score)
    ejecutar_script([sys.executable, os.path.join(dir_actual, "parte4_numba.py")], "PARTE 4: Numba (Aceleracion Numerica)")

    # 5. Parte AirLabs: descarga y cruce de vuelos con el clima (lluvia, visibilidad y retrasos)
    ejecutar_script([sys.executable, os.path.join(dir_actual, "parte_airlabs.py")], "PARTE AIRLABS: Trafico Aereo y Retrasos vs Clima")

    print("\n" + "=" * 65)
    print("pipeline ejecutado")
    print("Para abrir el Dashboard Web interactivo ejecuta:")
    print("  streamlit run app_web.py")


if __name__ == "__main__":
    # Punto de entrada principal para correr todas las etapas en orden
    main()
