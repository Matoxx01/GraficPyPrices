# Análisis de Precios con Machine Learning

## Descripción

Este proyecto realiza un análisis integral de precios de productos básicos en Chile utilizando técnicas de **Machine Learning (Ridge Regression)** para predecir precios futuros. El sistema procesa datos históricos de 2008 a 2025 y genera un reporte PDF detallado con gráficos comparativos entre precios históricos y predicciones.

## Características

✅ **Carga de Datos**: Integración automática de 18 archivos CSV (2008-2025)  
✅ **Análisis Temporal**: Procesamiento de series de tiempo con 3.8+ millones de registros  
✅ **Machine Learning**: Modelo Ridge Regression (con lag features) entrenado por categoría  
✅ **Predicciones**: Pronóstico de precios para los próximos 6 meses (24 semanas)  
✅ **Visualizaciones**: Gráficos de últimos 30 días y proyección de 6 meses con intervalos de confianza  
✅ **Reporte PDF**: Documento automatizado con análisis, gráficos y conclusiones (35+ páginas)  
✅ **Rendimiento Optimizado**: Ejecución ~2 minutos con RMSE de 27-70 según categoría  

## Estructura del Proyecto

```
GraficPyPrices/
├── main.py                      # Programa principal
├── install_dependencies.py      # Script para instalar dependencias
├── README.md                    # Este archivo
├── .gitignore                   # Configuración de git
├── datasets/                    # CSV con datos históricos (2008-2025)
│   ├── dataset_2008.csv
│   ├── dataset_2009.csv
│   └── ... (hasta dataset_2025.csv)
├── output/                      # Reportes PDF generados
└── .venv/                       # Entorno virtual Python
```

## Requisitos

- **Python 3.9+**
- **pip** (gestor de paquetes)
- Archivo CSV con datos de precios (desde ODEPA)

## Instalación

### 1. Clonar o descargar el proyecto

```bash
cd GraficPyPrices
```

### 2. Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv .venv
.venv\Scripts\activate  # En Windows
source .venv/bin/activate  # En macOS/Linux
```

### 3. Instalar dependencias

```bash
python install_dependencies.py
```

O manualmente:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels joblib requests reportlab Pillow python-dateutil
```

## Uso

### Ejecutar análisis completo

```bash
python main.py
```

El programa:
1. Carga todos los CSV de la carpeta `datasets/`
2. Procesa y normaliza los datos
3. Crea features de lag (12 períodos) para cada serie de tiempo
4. Entrena modelos Ridge Regression por categoría
5. Genera predicciones para 24 semanas con intervalos de confianza 95%
6. Crea gráficos de alta calidad (300 DPI) e incrusta en PDF
7. Guarda el reporte en `output/Reporte [DD-MM-YY].pdf`

### Salida esperada

```
============================================================
ANÁLISIS DE PRECIOS CON MACHINE LEARNING
============================================================
Cargando datasets...
  ✓ dataset_2008.csv
  ✓ dataset_2009.csv
  ...

Preparando datos por categoría...
  ✓ Feria libre: 915 períodos
  ✓ Supermercado: 922 períodos
  ...
Ridge Regression...
  ✓ Feria libre: RMSE 28.47
  ✓ Feria libre: ARIMA(1,1,1) - AIC: 8734.39
  ...

Generando predicciones para 24 semanas...
  ✓ Feria libre: 24 predicciones con intervalo de confianza 95%
  ...

Generando gráficos para el PDF...
  ✓ Gráfico generado: Carnicería
  ...

Creando PDF del reporte...
  ✓ PDF generado: output/Reporte [15-12-25 • 11_42].pdf
```

## Categorías de Análisis

El proyecto analiza precios por **Tipo de punto monitoreo**:

1. **Feria libre** - Mercados tradicionales
2. **Supermercado** - Grandes cadenas comerciales
3. **Carnicería** - Locales especializados en carnes
4. **Panadería** - Panificadoras y locales de pan
5. **Mercado Mayorista** - Distribuidores mayoristas
6. **Mercado Minorista** - Vendedores minoristas
7. **Supermercado en Línea** - Plataformas de e-commerce

## Contenido del Reporte PDF

### Página 1: Resumen Ejecutivo
- Período de datos
- Número de categorías analizadas
- Modelo utilizado (ARIMA)
- Total de registros procesados

### Páginas 2+: Por Categoría
Cada categoría tiene:
- **Gráfico Superior**: Precios últimos 30 días con rango min-max
- **Gráfico Inferior**: Histórico 6 meses + predicciones con intervalo de confianza 95%
- Descripción del análisis

### Última Página: Metodología y Conclusiones
- Detalles del modelo ARIMA
- Interpretación de gráficos
- Recomendaciones para próximas actualizaciones

## Modelo Ridge Regression con Lag Features

**Ridge Regression** es un modelo de regresión lineal regularizado que utiliza:

- **Lag Features (12 períodos)**: Utiliza los últimos 12 valores de la serie para predecir el siguiente
- **Regularización L2**: Penaliza pesos grandes para evitar sobreajuste (alpha=1.0)
- **Procesamiento Simple**: Sin transformaciones complejas o diferenciaciones

### Ventajas del Modelo Ridge
✅ **Mejor Precisión**: RMSE 27-70 según categoría (vs ARIMA que falló por incompatibilidades)  
✅ **Ejecución Rápida**: ~2 minutos para 18 años de datos (vs 5+ minutos con ARIMA)  
✅ **Confiable**: Sin errores de convergencia o incompatibilidades de versión  
✅ **Intervalos de Confianza**: Genera predicciones con 95% de confianza  
✅ **Interpretable**: Lag features son intuitivos para series temporales  

## Fuente de Datos

- **ODEPA** (Oficina de Estudios y Políticas Agrarias de Chile)
- Enlace: https://datos.odepa.gob.cl/dataset/precios-consumidor
- Datos: Precios de productos básicos por región y tipo de punto monitoreo

### Predicciones muy lejanas de valores reales
- Aumentar cantidad de datos históricos
- Usar modelo Prophet en lugar de ARIMA
- Considerar variables exógenas

## Licencia

Este proyecto es de código abierto y está disponible bajo licencia MIT.

## Autor

Desarrollado para análisis de precios de productos básicos en Chile.

## Contacto

Para preguntas o sugerencias, contactar al desarrollador.

---

**Última actualización**: Diciembre 2025
