# Proyecto 2. Predicción de Argumentos Efectivos

CC3084 Data Science, Universidad del Valle de Guatemala, Semestre II 2026.

Reto 19, Feedback Prize - Predicting Effective Arguments.
Integrantes: Genser Catalán y Javier Chávez.

El proyecto clasifica qué tan efectivo es un fragmento argumentativo dentro de un
ensayo escrito por estudiantes de secundaria. El conjunto trae 36,765 fragmentos
sacados de 4,191 ensayos, etiquetados en tres niveles de efectividad y siete tipos
de argumento. Primero se hizo un análisis exploratorio y después se entrenaron y
compararon cuatro modelos, que quedaron disponibles en una aplicación.

## Resultados

| Modelo | Log loss | F1 macro | Exactitud | Segundos |
|---|---|---|---|---|
| LightGBM | 0.6863 | 0.6020 | 0.6961 | 45.6 |
| DeBERTa | 0.7040 | 0.5890 | 0.6797 | 1,879.3 |
| TF-IDF con regresión logística | 0.7621 | 0.5463 | 0.6570 | 19.4 |
| Línea base | 0.9817 | 0.2404 | 0.5638 | 0.003 |

LightGBM es el mejor modelo. Le gana a DeBERTa porque es el único que recibe las
variables del ensayo completo, que en el análisis exploratorio mostraron una
correlación de 0.577 con la proporción de argumentos efectivos.

## Cómo correr el proyecto

**1. Clonar el repositorio y entrar a la carpeta**

```bash
git clone https://github.com/GenserDev/Proyecto2-DS.git && cd Proyecto2-DS
```

**2. Crear un entorno virtual y activarlo**

```bash
python3 -m venv .venv && source .venv/bin/activate
```

**3. Instalar las dependencias**

```bash
pip install -r app/requirements.txt
```

**4. Descargar los datos de Kaggle**

```bash
mkdir -p data/raw && kaggle competitions download -c feedback-prize-effectiveness -f train.csv -p data/raw
```

**5. Entrenar los modelos**

Abrir `notebooks/modelado-argumentos-efectivos.ipynb` y ejecutarlo completo.
Al terminar, copiar los archivos que genera a una carpeta `modelos` en la raíz.

**6. Levantar la aplicación**

```bash
streamlit run app/app.py
```

## Estructura

```
notebooks/   análisis exploratorio y modelado
app/         aplicación de Streamlit
informe/     informes de las dos fases
data/        datos de Kaggle, no se versiona
modelos/     modelos entrenados, no se versiona
```

## Notas

Los notebooks fueron desarrollados y ejecutados en Kaggle, por eso la variable
`ruta` apunta a `/kaggle/input/competitions/feedback-prize-effectiveness`. Si se
corren en local, hay que cambiarla por `data/raw`.

El bloque de DeBERTa necesita GPU y tarda cerca de una hora. Está controlado por
la bandera `ENTRENAR_TRANSFORMER` al inicio de la sección 10.

La aplicación no funciona sin los modelos entrenados. Si la carpeta `modelos` está
vacía, la pestaña de clasificación avisa que no los encontró.

Las celdas deben ejecutarse en orden, porque las secciones de modelado usan las
variables que se crean al inicio.
