import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from preparacion import construir_variables, cargar_artefactos, cargar_referencias, predecir

PALETA = {
    'Ineffective': '#C1495B',
    'Adequate': '#E0A458',
    'Effective': '#3E7C59',
    'principal': '#2F4B7C',
    'apoyo': '#8AA1C1'
}

CLASES = ['Ineffective', 'Adequate', 'Effective']

st.set_page_config(page_title='Argumentos Efectivos', page_icon='📝', layout='wide')


@st.cache_data
def cargar_datos():
    for ruta in ['../data/raw/train.csv', 'data/raw/train.csv', 'datos/train.csv']:
        if os.path.exists(ruta):
            return construir_variables(pd.read_csv(ruta))
    return None


@st.cache_data
def cargar_metricas():
    for ruta in ['modelos/metricas.json', '../modelos/metricas.json']:
        if os.path.exists(ruta):
            with open(ruta) as archivo:
                return json.load(archivo)
    return None


@st.cache_resource
def cargar_modelos():
    return cargar_artefactos()


@st.cache_data
def cargar_medianas():
    return cargar_referencias()


def seccion_clasificar(modelos, referencias):
    st.header('Clasificar un argumento')
    st.write('Escribe un fragmento argumentativo y elige qué función cumple dentro del ensayo.')

    columna_texto, columna_opciones = st.columns([3, 1])

    with columna_texto:
        texto = st.text_area('Texto del fragmento', height=170,
                             placeholder='Escribe aquí el argumento del estudiante')

    with columna_opciones:
        tipo = st.selectbox('Tipo de argumento',
                            ['Claim', 'Evidence', 'Position', 'Concluding Statement',
                             'Lead', 'Counterclaim', 'Rebuttal'])
        disponibles = list(modelos.keys()) if modelos else []
        elegidos = st.multiselect('Modelos', disponibles, default=disponibles)

    if not st.button('Clasificar', type='primary'):
        return

    if not texto.strip():
        st.warning('Escribe un texto antes de clasificar.')
        return

    if not elegidos:
        st.warning('Elige al menos un modelo.')
        return

    salidas = {nombre: predecir(modelos[nombre], texto, tipo, referencias) for nombre in elegidos}

    for nombre, probabilidades in salidas.items():
        st.subheader(nombre)
        clase = CLASES[int(np.argmax(probabilidades))]
        confianza = float(np.max(probabilidades))

        columna_resultado, columna_grafico = st.columns([1, 2])

        with columna_resultado:
            st.metric('Resultado', clase, f'{confianza:.1%} de confianza')

        with columna_grafico:
            figura = px.bar(x=CLASES, y=probabilidades,
                            color=CLASES, color_discrete_map=PALETA,
                            labels={'x': '', 'y': 'probabilidad'})
            figura.update_layout(showlegend=False, height=250,
                                 margin=dict(l=0, r=0, t=10, b=0), yaxis_range=[0, 1])
            st.plotly_chart(figura, use_container_width=True)

    if len(salidas) > 1:
        st.subheader('Comparación entre modelos')
        comparacion = pd.DataFrame(salidas, index=CLASES).T
        figura = px.imshow(comparacion, text_auto='.3f', aspect='auto',
                           color_continuous_scale=['#F2F2F2', PALETA['principal']])
        figura.update_layout(height=90 + 60 * len(salidas), coloraxis_showscale=False)
        st.plotly_chart(figura, use_container_width=True)


def seccion_datos(datos):
    st.header('Explorar los datos')

    if datos is None:
        st.info('No se encontró train.csv. Colócalo en data/raw para activar esta sección.')
        return

    filtro_tipo = st.multiselect('Filtrar por tipo de argumento',
                                 sorted(datos['discourse_type'].unique()))
    filtrados = datos[datos['discourse_type'].isin(filtro_tipo)] if filtro_tipo else datos

    fila = st.columns(4)
    fila[0].metric('Fragmentos', f'{len(filtrados):,}')
    fila[1].metric('Ensayos', f"{filtrados['essay_id'].nunique():,}")
    fila[2].metric('Palabras promedio', f"{filtrados['n_palabras'].mean():.1f}")
    fila[3].metric('Comas promedio', f"{filtrados['n_comas'].mean():.2f}")

    izquierda, derecha = st.columns(2)

    with izquierda:
        conteo = filtrados['discourse_effectiveness'].value_counts().reindex(CLASES)
        figura = px.bar(x=conteo.index, y=conteo.values, color=conteo.index,
                        color_discrete_map=PALETA, labels={'x': '', 'y': 'fragmentos'},
                        title='Distribución de efectividad')
        figura.update_layout(showlegend=False)
        st.plotly_chart(figura, use_container_width=True)

    with derecha:
        variable = st.selectbox('Variable numérica',
                                ['n_palabras', 'n_caracteres', 'n_comas',
                                 'largo_palabra', 'palabras_ensayo'])
        limite = filtrados[variable].quantile(0.98)
        figura = px.box(filtrados[filtrados[variable] <= limite],
                        x='discourse_effectiveness', y=variable,
                        color='discourse_effectiveness', color_discrete_map=PALETA,
                        category_orders={'discourse_effectiveness': CLASES},
                        title=f'{variable} según efectividad')
        figura.update_layout(showlegend=False, xaxis_title='')
        st.plotly_chart(figura, use_container_width=True)

    proporciones = pd.crosstab(filtrados['discourse_type'],
                               filtrados['discourse_effectiveness'],
                               normalize='index')[CLASES] * 100
    figura = px.bar(proporciones, orientation='h', color_discrete_map=PALETA,
                    labels={'value': 'porcentaje', 'discourse_type': ''},
                    title='Composición de efectividad por tipo de argumento')
    st.plotly_chart(figura, use_container_width=True)

    if st.checkbox('Ver la tabla de datos'):
        st.dataframe(filtrados.head(500), use_container_width=True)


def seccion_rendimiento(metricas):
    st.header('Rendimiento de los modelos')

    if metricas is None:
        st.info('No se encontró metricas.json. Corre el notebook de modelado para generarlo.')
        return

    if not st.checkbox('Mostrar el rendimiento', value=True):
        st.caption('Información oculta.')
        return

    tabla = pd.DataFrame(metricas).T.sort_values('log_loss')
    st.dataframe(tabla.style.format('{:.4f}'), use_container_width=True)

    izquierda, derecha = st.columns(2)

    with izquierda:
        figura = px.bar(tabla.sort_values('log_loss', ascending=False),
                        x='log_loss', orientation='h',
                        title='Log loss, menor es mejor')
        figura.update_traces(marker_color=PALETA['principal'])
        figura.update_layout(yaxis_title='')
        st.plotly_chart(figura, use_container_width=True)

    with derecha:
        figura = px.bar(tabla.sort_values('f1_macro'),
                        x='f1_macro', orientation='h',
                        title='F1 macro, mayor es mejor')
        figura.update_traces(marker_color=PALETA['Effective'])
        figura.update_layout(yaxis_title='')
        st.plotly_chart(figura, use_container_width=True)

    figura = go.Figure()
    for nombre in tabla.index:
        figura.add_trace(go.Scatter(
            x=[tabla.loc[nombre, 'segundos']], y=[tabla.loc[nombre, 'log_loss']],
            mode='markers+text', text=[nombre], textposition='top center',
            marker=dict(size=18), name=nombre))
    figura.update_layout(title='Costo de entrenamiento contra desempeño',
                         xaxis_title='segundos', yaxis_title='log loss', showlegend=False)
    st.plotly_chart(figura, use_container_width=True)


st.title('Predicción de Argumentos Efectivos')
st.caption('Reto 19, Feedback Prize. CC3084 Data Science, Universidad del Valle de Guatemala.')

modelos = cargar_modelos()
datos = cargar_datos()
metricas = cargar_metricas()
referencias = cargar_medianas()

if not modelos:
    st.warning('No se encontraron modelos entrenados. Corre el notebook de modelado y copia los '
               'archivos .joblib a la carpeta modelos.')

clasificar, explorar, rendimiento = st.tabs(['Clasificar', 'Explorar datos', 'Rendimiento'])

with clasificar:
    seccion_clasificar(modelos, referencias)

with explorar:
    seccion_datos(datos)

with rendimiento:
    seccion_rendimiento(metricas)
