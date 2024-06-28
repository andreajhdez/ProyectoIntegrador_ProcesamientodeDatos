from datasets import load_dataset
import numpy as np
import pandas as pd

# Cargar el dataset de insuficiencia cardíaca
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Convertir dataset a pandas
df = pd.DataFrame(data)

# Con este comando sabemos el tipo de datos de los elementos dentro del dataset
print(df.dtypes)

# Con las siguientes líneas de codigo convertiremos los datos a los correctos para cada categoría, al parecer todas están correctas, sin embargo se convertira la columna isdead a booleano
df['age'] = pd.to_numeric(df['age'])
df['creatinine_phosphokinase_concentration_in_blood'] = pd.to_numeric(
    df['creatinine_phosphokinase_concentration_in_blood'])
df['heart_ejection_fraction'] = pd.to_numeric(df['heart_ejection_fraction'])
df['platelets_concentration_in_blood'] = pd.to_numeric(
    df['platelets_concentration_in_blood'])
df['serum_creatinine_concentration_in_blood'] = pd.to_numeric(
    df['serum_creatinine_concentration_in_blood'])
df['serum_sodium_concentration_in_blood'] = pd.to_numeric(
    df['serum_sodium_concentration_in_blood'])
df['days_in_study'] = pd.to_numeric(df['days_in_study'])
df['is_dead'] = df['is_dead'].astype(bool)

# Calcular la cantidad de hombres fumadores vs mujeres fumadoras usando agregaciones en Pandas e imprimir estos valores
df_hombresfumadores = df[(df['is_male'] == 1) & (df['is_smoker'] == 1)]
df_mujeresfumadoras = df[(df['is_male'] == 0) & (df['is_smoker'] == 1)]

num_hombresfumadores = df_hombresfumadores.shape[0]
num_mujeresfumadoras = df_mujeresfumadoras.shape[0]

print(f'Número de Hombres Fumadores: {num_hombresfumadores}',
      f'| Número de Mujeres Fumadoras: {num_mujeresfumadoras}')
