import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.express as px

# Leer el archivo CSV
df = pd.read_csv('registros_ventas_limpios_con_categoria_edad.csv')

# Eliminar la columna 'categoria_edad' y la columna objetivo 'DEATH_EVENT'
X = df.drop(columns=['DEATH_EVENT', 'categoria_edad']).values
y = df['DEATH_EVENT'].values

X_embedded = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=3
).fit_transform(X)



# Convertir los datos a un DataFrame de Plotly
df_embedded = pd.DataFrame(X_embedded, columns=['Component 1', 'Component 2', 'Component 3'])
df_embedded['Death Event'] = y

print(df_embedded.head())   

# Crear el gráfico 3D
fig = px.scatter_3d(
    df_embedded, 
    x='Component 1', 
    y='Component 2', 
    z='Component 3', 
    color='Death Event',
    labels={'Death Event': 'Muerte'},
    title="Visualización 3D usando t-SNE"
)

# Añadir los colores y la leyenda
fig.update_layout(
    scene=dict(
        xaxis_title='Componente 1',
        yaxis_title='Componente 2',
        zaxis_title='Componente 3'
    ),
    legend=dict(
        x=0,
        y=1,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='Black',
        borderwidth=2
    )
)

# Mostrar el gráfico
fig.show()