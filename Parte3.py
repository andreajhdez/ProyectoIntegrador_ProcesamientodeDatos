from datasets import load_dataset
import numpy as np
import pandas as pd

# Cargar el dataset de insuficiencia cardíaca
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Convertir dataset a pandas
df = pd.DataFrame(data)
print(df.dtypes)

# Separar df en dos diferentes, uno para fallecidos y otro para vivos
df_fallecidos = df[df['is_dead'] == 1]
df_vivos = df[df['is_dead'] == 0]

# Calcular promedios de edades de los distintos df
prom_fallecidos = df_fallecidos['age'].mean()
prom_vivos = df_vivos['age'].mean()

print(f'Edad Promedio fallecidos: {round(prom_fallecidos,2)}',
      f'| Edad Promedio NO fallecidos (vivos): {round(prom_vivos,2)} )')