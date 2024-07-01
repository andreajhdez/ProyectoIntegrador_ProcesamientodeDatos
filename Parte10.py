import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go

def prediccion_edades() -> tuple:
    # Cargar el dataset
    df = pd.read_csv('registros_ventas_limpios_con_categoria_edad.csv')

    # Definir las variables X y y
    X = df.drop(columns=['DEATH_EVENT', 'age', 'categoria_edad'])
    y = df['age']

    # Ajustar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Realizar las predicciones
    y_pred = model.predict(X)

    # Calcular el error cuadrático medio
    mse = mean_squared_error(y, y_pred)

    # Mostrar resultados
    resultados = pd.DataFrame({'Edad Real': y, 'Edad Predicha': y_pred})
    print(resultados)
    print("Error cuadrático medio de las predicciones de edad:", round(mse, 2))

    # Guardar resultados como CSV
    resultados.to_csv('predicciones_edad_regresionlineal.csv', index=False)

    return mse

prediccion_edades()
