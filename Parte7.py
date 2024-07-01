import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer el archivo CSV generado por el script de ETL
df = pd.read_csv('registros_ventas_limpios_con_categoria_edad.csv')

# Calcular el número de bins usando la Regla de Sturges
N = len(df['age'])
num_bins = int(1 + np.log2(N))
print(f"Número de bins calculado con la Regla de Sturges: {num_bins}")

# Graficar la distribución de edades con un histograma
plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=num_bins, edgecolor='k', alpha=0.7)
plt.title('Distribución de Edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.grid(True)

# Mostrar la gráfica y guardar en png la imagen
plt.savefig('distribucion_edades.png', dpi=300)
plt.show()

data = pd.read_csv('registros_ventas_limpios_con_categoria_edad.csv')

df = pd.DataFrame(data)

# Agrupar por sexo y contar los valores de cada categoría
grouped = df.groupby('sex').agg({
    'anaemia': 'sum',
    'diabetes': 'sum',
    'smoking': 'sum',
    'DEATH_EVENT': 'sum'
})

# Preparar datos para graficar
categories = ['Anémicos', 'Diabéticos', 'Fumadores', 'Muertos']
men_counts = grouped.loc[1].values
women_counts = grouped.loc[0].values

# Configurar colores
colors = ['blue', 'blue', 'blue', 'blue']  # Hombres en azul
women_colors = ['red', 'red', 'red', 'red']  # Mujeres en rojo

# Crear la figura y los ejes
fig, ax = plt.subplots()

# Definir el ancho de las barras
bar_width = 0.4

# Graficar barras para hombres (primero)
ax.bar(categories, men_counts, width=-bar_width,
       align='edge', color=colors, label='Hombres')

# Graficar barras para mujeres (luego)
ax.bar(categories, women_counts, width=bar_width,
       align='edge', color=women_colors, label='Mujeres')

# Configurar título y etiquetas de ejes
plt.title('Histograma Agrupado por Sexo')
plt.xlabel('Categorías')
plt.ylabel('Cantidad')

# Mostrar leyenda
plt.legend()

# Mostrar la gráfica y guardar en png la imagen
plt.savefig('histograma_agrupado_sexo.png', dpi=300)
plt.show()
pass