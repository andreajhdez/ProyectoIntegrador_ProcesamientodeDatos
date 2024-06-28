from datasets import load_dataset
import numpy as np

# Cargar el dataset de insuficiencia cardíaca
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Convertir la lista de edades a un arreglo de NumPy
lista_edades = np.array(data["age"])

# Calcular el promedio de edad
promedio_edad = np.mean(lista_edades)

print(f"El promedio de edad de los pacientes es: {promedio_edad}")
print(f"El array de edades es: {lista_edades}")
