import sys
import requests
import pandas as pd

# Esta función sirve para descargar los datos de la url


def descargar_url(url):
    response = requests.get(url)
    contenido = response.content
    with open('datos1.csv', 'wb') as archivo:
        archivo.write(contenido)


def convertir_a_dataframe(data):
    # Convertir los datos a DataFrame
    # Ajusta el separador según el formato de los datos
    df = pd.read_csv(data, sep=';')
    return df


def limpieza_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Verificar valores faltantes
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

# función con clasificación por edades


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

# Se crea columna que categoriza por edades


def crear_columna(df):
    # Se crea una columna utilizando la función con la condición que aplica para las distintas edades.
    df['categoria_edad'] = df['age'].apply(columna_edades)
    return df

# Se guarda el resultado como csv


def guardar_csv(df):
    df.to_csv('registros_ventas_limpios_con_categoria_edad.csv', index=False)
    return df


def main(url):
    # Descargar datos desde la URL
    descargar_url(url)

    # Convertir los datos a DataFrame
    df = convertir_a_dataframe('datos1.csv')

    # Limpiar el DataFrame
    df = limpieza_dataframe(df)

    # Crear columna de categoría de edad
    df = crear_columna(df)

    # Guardar el DataFrame limpio y categorizado como CSV
    guardar_csv(df)
    print("Proceso completado: datos limpios y categorizados guardados como 'registros_ventas_limpios_con_categoria_edad.csv'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    main(url)
