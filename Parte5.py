import requests
import pandas as pd
import numpy as np

# Función reutilizable que usa un get request para descargar los datos de una url y escribe la respuesta como un archivo de texto plano con extensión csv
def descargar_csv(url):
    response = requests.get(url)
    contenido = response.content
    with open('datos.csv', 'wb') as archivo:
        archivo.write(contenido)
    df = pd.read_csv('datos.csv')
    return df 

def limpieza_datos(df: pd.DataFrame) -> pd.DataFrame:
    # Punto 1: Verificar valores faltantes
    valores_faltantes = df.isna().sum()
    print("Valores faltantes antes de la limpieza: ")
    print(valores_faltantes)
  
    # Manejar valores faltantes
    if valores_faltantes.any():
        print("Hay valores faltantes, se procederá a rellenarlos.")
        df.fillna(df.mean(), inplace=True)

    # Punto 2: Verificar filas duplicadas
    filas_duplicadas = df.duplicated().sum()
    print(f"Filas duplicadas antes de la limpieza: {filas_duplicadas}")
    
    # Eliminar filas duplicadas
    if filas_duplicadas.any():
        df.drop_duplicates(inplace=True)
    
    # Punto 3: Eliminar valores atípicos usando el rango intercuartil (IQR)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

    return df
#función con clasificación por edades
def columna_edades(age):
    if age <= 12:
        return 'Niño'
    elif 13 <= age <= 19:
        return 'Adolescente'
    elif 20 <= age <= 39:
        return 'Joven adulto'
    elif 40 <= age <= 59:
        return 'Adulto'
    else:
        return 'Adulto mayor'
    
# Punto 4: Se crea columna que categoriza por edades
def crear_columna(df):
    #Se crea una columna utilizando la función con la condición que aplica para las distintas edades.
    df['categoria_edad'] = df['age'].apply(columna_edades)
    return df

# Punto 5: se guarda el resultado como csv
def guardar_csv(df):
    df.to_csv('registros_ventas_limpios_con_categoria_edad.csv', index=False)
    return df
    
# Se activa la función con la url correspondiente para descargar los datos
url_proyectointegrador = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
df = descargar_csv(url_proyectointegrador)

#Se limpian los datos con la función de limpieza
df_limpio = limpieza_datos(df)
#Se crea columna con categoría edad
df_limpio_categoriaedad = crear_columna(df_limpio)
#Se guarda df limpio con la columna de categoría edad en un nuevo csv.
guardar_csv(df_limpio_categoriaedad)