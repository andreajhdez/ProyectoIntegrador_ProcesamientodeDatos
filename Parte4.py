import requests

# Función reutilizable que usa un get request para descargar los datos de una url y escribe la respuesta como un archivo de texto plano con extensión csv


def descargar_csv(url):
    response = requests.get(url)
    contenido = response.content
    with open('datos.csv', 'wb') as archivo:
        archivo.write(contenido)


# Se activa la función con la url correspondiente para descargar los datos
url_proyectointegrador = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
descargar_csv(url_proyectointegrador)
