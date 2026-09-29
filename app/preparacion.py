import json
import os

import joblib
import numpy as np
import pandas as pd

RUTAS = ['modelos', '../modelos', 'app/modelos']


def construir_variables(tabla):
    tabla = tabla.copy()
    tabla['discourse_text'] = tabla['discourse_text'].str.strip()
    tabla['n_caracteres'] = tabla['discourse_text'].str.len()
    tabla['n_palabras'] = tabla['discourse_text'].str.split().str.len()
    tabla['n_oraciones'] = tabla['discourse_text'].str.count(r'[.!?]').clip(lower=1)
    tabla['n_comas'] = tabla['discourse_text'].str.count(',')
    tabla['largo_palabra'] = tabla['n_caracteres'] / tabla['n_palabras']
    tabla['palabras_por_oracion'] = tabla['n_palabras'] / tabla['n_oraciones']
    tabla['log_palabras'] = np.log1p(tabla['n_palabras'])

    tabla['posicion'] = tabla.groupby('essay_id').cumcount()
    tabla['total_fragmentos'] = tabla.groupby('essay_id')['discourse_id'].transform('count')
    tabla['posicion_relativa'] = (tabla['posicion'] / (tabla['total_fragmentos'] - 1)).fillna(0.5)

    tabla['palabras_ensayo'] = tabla.groupby('essay_id')['n_palabras'].transform('sum')
    tabla['tipos_ensayo'] = tabla.groupby('essay_id')['discourse_type'].transform('nunique')
    tabla['proporcion_del_ensayo'] = tabla['n_palabras'] / tabla['palabras_ensayo']
    tabla['palabras_vs_tipo'] = tabla['n_palabras'] / tabla.groupby('discourse_type')['n_palabras'].transform('median')

    tabla['tipo_anterior'] = tabla.groupby('essay_id')['discourse_type'].shift(1).fillna('INICIO')
    tabla['tipo_siguiente'] = tabla.groupby('essay_id')['discourse_type'].shift(-1).fillna('FIN')
    return tabla


def buscar(nombre):
    for carpeta in RUTAS:
        ruta = os.path.join(carpeta, nombre)
        if os.path.exists(ruta):
            return ruta
    return None


def cargar_referencias():
    ruta = buscar('referencias.json')
    if not ruta:
        return None
    with open(ruta) as archivo:
        return json.load(archivo)


def cargar_artefactos():
    modelos = {}

    vectorizador = buscar('vectorizador.joblib')
    regresion = buscar('regresion.joblib')
    if vectorizador and regresion:
        modelos['TF-IDF + Regresión logística'] = {
            'tipo': 'tfidf',
            'vectorizador': joblib.load(vectorizador),
            'modelo': joblib.load(regresion)
        }

    bosque = buscar('lightgbm.joblib')
    if bosque and vectorizador:
        artefacto = joblib.load(bosque)
        artefacto['tipo'] = 'lightgbm'
        artefacto['vectorizador'] = joblib.load(vectorizador)
        modelos['LightGBM'] = artefacto

    return modelos


def predecir(artefacto, texto, tipo, referencias=None):
    entrada = pd.DataFrame([{
        'discourse_id': 'nuevo',
        'essay_id': 'nuevo',
        'discourse_text': texto,
        'discourse_type': tipo
    }])
    entrada = construir_variables(entrada)
    entrada['texto_completo'] = entrada['discourse_type'] + ' [SEP] ' + entrada['discourse_text']

    if referencias:
        for columna, valor in referencias['mediana_ensayo'].items():
            entrada[columna] = valor
        mediana_tipo = referencias['mediana_por_tipo'].get(tipo)
        if mediana_tipo:
            entrada['palabras_vs_tipo'] = entrada['n_palabras'] / mediana_tipo

    matriz_texto = artefacto['vectorizador'].transform(entrada['texto_completo'])

    if artefacto['tipo'] == 'tfidf':
        return artefacto['modelo'].predict_proba(matriz_texto)[0]

    numericas = ['n_caracteres', 'n_palabras', 'n_oraciones', 'n_comas', 'largo_palabra',
                 'palabras_por_oracion', 'log_palabras', 'posicion', 'total_fragmentos',
                 'posicion_relativa', 'palabras_ensayo', 'tipos_ensayo',
                 'proporcion_del_ensayo', 'palabras_vs_tipo']
    categoricas = ['discourse_type', 'tipo_anterior', 'tipo_siguiente']

    tabla = np.hstack([
        artefacto['escalador'].transform(entrada[numericas]),
        artefacto['codificador'].transform(entrada[categoricas]),
        artefacto['reductor'].transform(matriz_texto)
    ])
    return artefacto['modelo'].predict_proba(tabla)[0]
