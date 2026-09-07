# Proyecto 2. Análisis Exploratorio

CC3084 Data Science, Universidad del Valle de Guatemala, Semestre II 2026.

Análisis exploratorio del reto 19, Feedback Prize - Predicting Effective Arguments.
El conjunto trae 36,765 fragmentos argumentativos sacados de 4,191 ensayos de
estudiantes de secundaria, cada uno etiquetado con su tipo de argumento y con qué
tan efectivo es. El objetivo del análisis es entender qué distingue a un argumento
efectivo de uno que no lo es, antes de construir cualquier modelo.

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
pip install pandas numpy matplotlib seaborn scipy jupyter
```

**4. Descargar los datos de Kaggle**

```bash
kaggle competitions download -c feedback-prize-effectiveness -p data/raw && unzip data/raw/feedback-prize-effectiveness.zip -d data/raw
```

**5. Abrir y ejecutar el notebook**

```bash
jupyter notebook notebooks/eda-argumentos-efectivos.ipynb
```

## Notas

El notebook fue desarrollado y ejecutado en Kaggle, por eso la variable `ruta`
apunta a `/kaggle/input/competitions/feedback-prize-effectiveness`. Si se corre
local, hay que cambiarla por `data/raw`.

Correr desde Kaggle evita la descarga por completo, los datos ya vienen montados
al crear el notebook desde la página de la competencia.

Las celdas deben ejecutarse en orden, porque las secciones 4 en adelante usan las
variables numéricas que se crean en la celda de derivación.

La carpeta `data/` está en el `.gitignore` y no se sube al repositorio.
