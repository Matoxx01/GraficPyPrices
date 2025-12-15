import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning - Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# PDF
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Web scraping
import requests
from io import StringIO
import json
import tempfile
import shutil

# Configuración
DATASETS_PATH = 'datasets'
OUTPUT_PATH = 'output'
MODELS_PATH = 'models'

# Crear directorios si no existen
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

def format_fecha_español(fecha):
    """Formatear fecha a español"""
    meses = {
        'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
        'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
        'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
        'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
    }
    fecha_formateada = fecha.strftime("%d de %B de %Y, %H:%M")
    for mes_ing, mes_esp in meses.items():
        fecha_formateada = fecha_formateada.replace(mes_ing, mes_esp)
    return fecha_formateada

class PreciosAnalyzer:
    """Clase para analizar precios de productos básicos con Ridge Regression"""
    
    def __init__(self):
        self.data = None
        self.data_odepa = None
        self.models = {}
        self.predictions = {}
        self.temp_dir = None
        
    def cargar_datasets(self):
        """Cargar todos los CSV de la carpeta datasets"""
        print("Cargando datasets...")
        dataframes = []
        
        dataset_files = sorted(Path(DATASETS_PATH).glob('dataset_*.csv'))
        
        for file in dataset_files:
            try:
                df = pd.read_csv(file, encoding='utf-8')
                dataframes.append(df)
                print(f"  ✓ {file.name}")
            except Exception as e:
                print(f"  ✗ Error cargando {file.name}: {e}")
        
        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
            self._procesar_datos()
            print(f"Total de registros cargados: {len(self.data)}")
        else:
            print("No se encontraron datasets")
    
    def _procesar_datos(self):
        """Procesar y limpiar datos"""
        # Convertir fechas
        self.data['Fecha inicio'] = pd.to_datetime(self.data['Fecha inicio'])
        self.data['Fecha termino'] = pd.to_datetime(self.data['Fecha termino'])
        
        # Convertir precio promedio (cambia coma por punto)
        self.data['Precio promedio'] = self.data['Precio promedio'].astype(str).str.replace(',', '.').astype(float)
        self.data['Precio minimo'] = self.data['Precio minimo'].astype(float)
        self.data['Precio maximo'] = self.data['Precio maximo'].astype(float)
        
        # Crear fecha central
        self.data['Fecha'] = self.data['Fecha inicio'] + (self.data['Fecha termino'] - self.data['Fecha inicio']) / 2
        
        # Ordenar por fecha
        self.data = self.data.sort_values('Fecha').reset_index(drop=True)
    
    def descargar_csvs_odepa(self):
        """Descargar automáticamente los CSV más recientes desde ODEPA"""
        print("\nActualizando datos desde ODEPA...")
        try:
            # URL de la página de ODEPA con los datasets
            url_base = "https://datos.odepa.gob.cl/dataset/precios-consumidor"
            
            # Intentar obtener la lista de recursos disponibles
            api_url = "https://datos.odepa.gob.cl/api/3/action/package_show"
            params = {'id': 'precios-consumidor'}
            
            response = requests.get(api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and 'resources' in data['result']:
                    resources = data['result']['resources']
                    
                    # Buscar recursos CSV
                    csvs_descargados = 0
                    for resource in resources:
                        try:
                            if 'csv' in resource.get('format', '').lower():
                                nombre = resource.get('name', '')
                                url_descarga = resource.get('url', '')
                                
                                # Verificar si es un CSV de datos de precios
                                if url_descarga and ('precios' in nombre.lower() or 'datos' in nombre.lower()):
                                    # Descargar CSV
                                    csv_response = requests.get(url_descarga, timeout=15)
                                    if csv_response.status_code == 200:
                                        # Guardar en carpeta datasets
                                        archivo_local = os.path.join(DATASETS_PATH, f"{nombre.split('.')[0]}.csv")
                                        with open(archivo_local, 'wb') as f:
                                            f.write(csv_response.content)
                                        csvs_descargados += 1
                                        print(f"  ✓ Descargado: {nombre}")
                        except Exception as e:
                            pass  # Continuar con siguiente recurso
                    
                    if csvs_descargados > 0:
                        print(f"  ✓ {csvs_descargados} archivo(s) actualizado(s) desde ODEPA")
                        return True
        except Exception as e:
            print(f"  ⚠ No se pudo actualizar desde ODEPA: {e}")
        
        print("  Continuando con datos locales disponibles...")
        return False
    
    def preparar_datos_por_categoria(self):
        """Preparar datos agrupados por Tipo de punto monitoreo"""
        print("\nPreparando datos por categoría...")
        
        categorias_data = {}
        
        for categoria in self.data['Tipo de punto monitoreo'].unique():
            datos_cat = self.data[self.data['Tipo de punto monitoreo'] == categoria].copy()
            
            # Agrupar por fecha y obtener promedio de precios
            datos_agrupados = datos_cat.groupby('Fecha').agg({
                'Precio promedio': 'mean',
                'Precio minimo': 'mean',
                'Precio maximo': 'mean'
            }).reset_index()
            
            datos_agrupados = datos_agrupados.sort_values('Fecha')
            categorias_data[categoria] = datos_agrupados
            
            print(f"  ✓ {categoria}: {len(datos_agrupados)} períodos")
        
        return categorias_data
    
    def entrenar_modelos(self, categorias_data):
        """Entrenar modelos Gradient Boosting para cada categoría"""
        print("\nEntrenando modelos Gradient Boosting...")
        
        for categoria, datos in categorias_data.items():
            try:
                serie = datos['Precio promedio'].values
                
                # Solo entrenar si hay suficientes datos
                if len(serie) < 30:
                    print(f"  ⚠ {categoria}: Datos insuficientes para entrenar")
                    continue
                
                # Crear features avanzadas (lag + rolling mean + tendencia + volatilidad)
                lag = 12
                X = []
                y = []
                
                for i in range(lag, len(serie)):
                    # Features básicos (lag)
                    features = list(serie[i-lag:i])
                    
                    # Media móvil (últimas 4 semanas)
                    features.append(np.mean(serie[max(0, i-4):i]))
                    
                    # Diferencia (cambio desde semana anterior)
                    features.append(serie[i-1] - serie[i-2] if i >= 2 else 0)
                    
                    # Tendencia (últimas 4 semanas)
                    if i >= 4:
                        tendencia = (serie[i-1] + serie[i-2] + serie[i-3] + serie[i-4]) / 4 - np.mean(serie[max(0, i-8):i-4])
                        features.append(tendencia)
                    else:
                        features.append(0)
                    
                    # Volatilidad
                    features.append(np.std(serie[max(0, i-4):i]))
                    
                    X.append(features)
                    y.append(serie[i])
                
                X = np.array(X)
                y = np.array(y)
                
                # Entrenar modelo Gradient Boosting
                modelo = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                modelo.fit(X, y)
                
                # Calcular RMSE
                y_pred = modelo.predict(X)
                rmse = np.sqrt(np.mean((y - y_pred)**2))
                
                self.models[categoria] = {
                    'modelo': modelo,
                    'serie_completa': serie,
                    'rmse': rmse,
                    'lag': lag
                }
                
                print(f"  ✓ {categoria}: Gradient Boosting - RMSE: {rmse:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error en {categoria}: {e}")
    
    def generar_predicciones(self, categorias_data, meses_adelante=6):
        """Generar predicciones para los próximos meses"""
        print(f"\nGenerando predicciones para {meses_adelante} meses...")
        
        predicciones = {}
        
        for categoria, modelo_info in self.models.items():
            try:
                datos = categorias_data[categoria]
                fecha_ultima = datos['Fecha'].max()
                modelo = modelo_info['modelo']
                serie_completa = modelo_info['serie_completa']
                lag = modelo_info['lag']
                rmse = modelo_info['rmse']
                
                # Generar predicciones
                steps = meses_adelante * 4  # Aproximadamente 4 semanas por mes
                
                # Usar los últimos valores como entrada inicial
                secuencia_actual = serie_completa[-lag:].copy()
                predicciones_list = []
                
                # Predicción iterativa con features avanzadas
                for _ in range(steps):
                    # Crear features igual como en el entrenamiento
                    features = list(secuencia_actual)
                    
                    # Media móvil (últimas 4 semanas)
                    features.append(np.mean(secuencia_actual[-4:]))
                    
                    # Diferencia (cambio desde semana anterior)
                    features.append(secuencia_actual[-1] - secuencia_actual[-2] if len(secuencia_actual) >= 2 else 0)
                    
                    # Tendencia (últimas 4 semanas)
                    if len(secuencia_actual) >= 4:
                        tendencia = (secuencia_actual[-1] + secuencia_actual[-2] + secuencia_actual[-3] + secuencia_actual[-4]) / 4 - np.mean(secuencia_actual[-8:-4] if len(secuencia_actual) >= 8 else secuencia_actual[:-4])
                        features.append(tendencia)
                    else:
                        features.append(0)
                    
                    # Volatilidad
                    features.append(np.std(secuencia_actual[-4:]))
                    
                    # Predecir siguiente valor
                    next_pred = modelo.predict(np.array([features]))[0]
                    predicciones_list.append(next_pred)
                    
                    # Actualizar secuencia
                    secuencia_actual = np.append(secuencia_actual[1:], next_pred)
                
                pred_mean = np.array(predicciones_list)
                
                # Calcular intervalo de confianza basado en RMSE histórico
                lower_ci = pred_mean - (1.96 * rmse)
                upper_ci = pred_mean + (1.96 * rmse)
                
                # Crear fechas futuras
                fechas_futuras = pd.date_range(start=fecha_ultima + timedelta(days=7), 
                                               periods=len(pred_mean), freq='7D')
                
                predicciones[categoria] = {
                    'fechas': fechas_futuras,
                    'predicciones': pred_mean,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci,
                    'datos_historicos': datos,
                    'rmse': rmse
                }
                
                print(f"  ✓ {categoria}: {len(pred_mean)} predicciones")
            except Exception as e:
                print(f"  ✗ Error prediciendo {categoria}: {e}")
        
        self.predictions = predicciones
        return predicciones
    
    def format_fecha_grafico(self, fecha):
        """Formatear fecha para gráficos en formato Mes-Día"""
        meses = {
            'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
            'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
            'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
            'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
        }
        fecha_obj = pd.Timestamp(fecha)
        mes_ing = fecha_obj.strftime('%B')
        mes_esp = meses.get(mes_ing, mes_ing)
        dia = fecha_obj.strftime('%d').lstrip('0')  # Eliminar cero inicial
        return f"{mes_esp}-{dia}"
    
    def crear_formateador_fecha(self):
        """Crear formateador personalizado para fechas"""
        from matplotlib.ticker import FuncFormatter
        def formato_fecha(x, pos):
            fecha = mdates.num2date(x)
            return self.format_fecha_grafico(fecha)
        return FuncFormatter(formato_fecha)
    
    def calcular_ultimos_30_dias(self, categorias_data):
        """Obtener datos de últimos 30 días"""
        ultimos_30 = {}
        
        for categoria, datos in categorias_data.items():
            fecha_max = datos['Fecha'].max()
            fecha_inicio = fecha_max - timedelta(days=30)
            
            datos_30 = datos[(datos['Fecha'] >= fecha_inicio) & (datos['Fecha'] <= fecha_max)]
            ultimos_30[categoria] = datos_30
        
        return ultimos_30
    
    def generar_grafico_general(self, categorias_data):
        """Generar gráfico general de precios promedio trimestral"""
        print("Generando gráfico general trimestral...")
        
        try:
            # Combinar todos los datos
            todos_datos = pd.DataFrame()
            for categoria, datos in categorias_data.items():
                todos_datos = pd.concat([todos_datos, datos[['Fecha', 'Precio promedio']]], ignore_index=True)
            
            # Agrupar por fecha y calcular promedio general
            datos_promedio = todos_datos.groupby('Fecha')['Precio promedio'].mean().reset_index()
            datos_promedio = datos_promedio.sort_values('Fecha')
            
            # Agrupar por trimestre (cada 3 meses)
            datos_promedio['Trimestre'] = datos_promedio['Fecha'].dt.to_period('Q')
            datos_trimestral = datos_promedio.groupby('Trimestre')['Precio promedio'].agg(['mean', 'min', 'max']).reset_index()
            datos_trimestral['Trimestre_str'] = datos_trimestral['Trimestre'].astype(str)
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Gráfico de línea con área
            ax.plot(range(len(datos_trimestral)), datos_trimestral['mean'], 
                   marker='o', linestyle='-', linewidth=2.5, label='Precio Promedio', 
                   color='#2E86AB', markersize=6)
            
            # Área de rango (min-max)
            ax.fill_between(range(len(datos_trimestral)), 
                           datos_trimestral['min'], 
                           datos_trimestral['max'], 
                           alpha=0.2, color='#2E86AB', label='Rango (Min-Max)')
            
            # Configuración - Eje X por años
            ax.set_xlabel('Año', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precio Promedio ($)', fontsize=12, fontweight='bold')
            ax.set_title('Evolución de Precios Promedio por Trimestre (2008-2025)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Obtener posiciones de cambio de año
            años = datos_trimestral['Trimestre'].dt.year.unique()
            posiciones_años = []
            etiquetas_años = []
            
            for año in años:
                idx_primer_trimestre = datos_trimestral[datos_trimestral['Trimestre'].dt.year == año].index.min()
                posiciones_años.append(idx_primer_trimestre)
                etiquetas_años.append(str(año))
            
            ax.set_xticks(posiciones_años)
            ax.set_xticklabels(etiquetas_años, rotation=45, ha='right')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Mejorar apariencia
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Guardar en archivo temporal
            grafico_path = os.path.join(self.temp_dir, 'grafico_general_trimestral.png')
            plt.savefig(grafico_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Gráfico general generado: {len(datos_trimestral)} trimestres")
            return grafico_path
            
        except Exception as e:
            print(f"  ✗ Error generando gráfico general: {e}")
            return None
        """Obtener datos de últimos 30 días"""
        ultimos_30 = {}
        
        for categoria, datos in categorias_data.items():
            fecha_max = datos['Fecha'].max()
            fecha_inicio = fecha_max - timedelta(days=30)
            
            datos_30 = datos[(datos['Fecha'] >= fecha_inicio) & (datos['Fecha'] <= fecha_max)]
            ultimos_30[categoria] = datos_30
        
        return ultimos_30
    
    def generar_graficos_pdf(self, categorias_data):
        """Generar gráficos y crear PDF del reporte"""
        print("\nGenerando gráficos para PDF...")
        
        # Crear directorio temporal para imágenes
        self.temp_dir = tempfile.mkdtemp()
        
        # Configurar matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        graficos_paths = []
        
        # Generar gráfico general trimestral
        grafico_general = self.generar_grafico_general(categorias_data)
        
        # Procesar cada categoría
        for categoria in sorted(self.predictions.keys()):
            try:
                pred_data = self.predictions[categoria]
                
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                fig.suptitle(f'Análisis de Precios - {categoria}', fontsize=16, fontweight='bold')
                
                # Gráfico 1: Últimos 30 días
                ax1 = axes[0]
                datos_30 = self.calcular_ultimos_30_dias(categorias_data)[categoria]
                
                ax1.plot(datos_30['Fecha'], datos_30['Precio promedio'], 
                        marker='o', linestyle='-', linewidth=2, label='Precio Histórico', color='#2E86AB')
                ax1.fill_between(datos_30['Fecha'], 
                                datos_30['Precio minimo'], 
                                datos_30['Precio maximo'], 
                                alpha=0.2, color='#2E86AB', label='Rango (Min-Max)')
                ax1.set_title(f'Precios Últimos 30 Días - {categoria}', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Precio ($)', fontsize=11)
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(self.crear_formateador_fecha())
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                
                # Gráfico 2: Predicciones vs Histórico (6 meses) + Backtest (3 meses pasados)
                ax2 = axes[1]
                
                # Datos históricos (6 meses)
                fecha_inicio_6m = pred_data['fechas'][0] - timedelta(days=180)
                datos_6m = pred_data['datos_historicos'][
                    pred_data['datos_historicos']['Fecha'] >= fecha_inicio_6m
                ]
                
                ax2.plot(datos_6m['Fecha'], datos_6m['Precio promedio'], 
                        marker='o', linestyle='-', linewidth=2.5, label='Histórico Real', color='#2E86AB')
                
                # Generar predicciones retroactivas para todo el período histórico mostrado (validación)
                serie_completa = pred_data['datos_historicos']['Precio promedio'].values
                fechas_historicas = pred_data['datos_historicos']['Fecha'].values
                lag = 12
                
                # Encontrar punto de inicio para predicciones retroactivas (mismo inicio que período histórico mostrado)
                fecha_inicio_6m = pred_data['fechas'][0] - timedelta(days=180)
                idx_inicio_backtest = None
                for i, fecha in enumerate(fechas_historicas):
                    if pd.Timestamp(fecha) >= fecha_inicio_6m:
                        idx_inicio_backtest = i
                        break
                
                if idx_inicio_backtest is not None and idx_inicio_backtest >= lag:
                    predicciones_backtest = []
                    fechas_backtest = []
                    
                    # Hacer predicciones para cada punto en todo el período histórico
                    for i in range(idx_inicio_backtest, len(serie_completa)):
                        if i >= lag:
                            # Usar los 12 períodos anteriores para crear features
                            ventana = serie_completa[i-lag:i]
                            
                            # Crear features igual como en el entrenamiento
                            features = list(ventana)
                            features.append(np.mean(ventana[-4:]))
                            features.append(ventana[-1] - ventana[-2] if len(ventana) >= 2 else 0)
                            
                            if len(ventana) >= 4:
                                tendencia = (ventana[-1] + ventana[-2] + ventana[-3] + ventana[-4]) / 4 - np.mean(ventana[-8:-4] if len(ventana) >= 8 else ventana[:-4])
                                features.append(tendencia)
                            else:
                                features.append(0)
                            
                            features.append(np.std(ventana[-4:]))
                            
                            pred = self.models[categoria]['modelo'].predict(np.array([features]))[0]
                            predicciones_backtest.append(pred)
                            fechas_backtest.append(fechas_historicas[i])
                    
                    if predicciones_backtest:
                        ax2.plot(fechas_backtest, predicciones_backtest, 
                                marker='^', linestyle=':', linewidth=2, label='Predicción Modelo (período histórico)', 
                                color='#F18F01', alpha=0.8)
                
                # Predicciones futuras
                ax2.plot(pred_data['fechas'], pred_data['predicciones'], 
                        marker='s', linestyle='--', linewidth=2.5, label='Predicción Futura', color='#A23B72')
                
                # Intervalo de confianza (solo futuro)
                ax2.fill_between(pred_data['fechas'], 
                                pred_data['lower_ci'], 
                                pred_data['upper_ci'], 
                                alpha=0.2, color='#A23B72', label='Intervalo 95%')
                
                # Línea vertical separando pasado y futuro
                ax2.axvline(x=pred_data['fechas'][0], color='red', linestyle=':', linewidth=1.5, alpha=0.5)
                
                ax2.set_title(f'Validación del Modelo + Predicción Futura (6 meses) - {categoria}', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Precio ($)', fontsize=11)
                ax2.set_xlabel('Fecha', fontsize=11)
                ax2.legend(loc='best', fontsize=9)
                ax2.grid(True, alpha=0.3)
                ax2.xaxis.set_major_formatter(self.crear_formateador_fecha())
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                
                plt.tight_layout()
                
                # Guardar gráfico en archivo temporal
                grafico_path = os.path.join(self.temp_dir, f'grafico_{categoria.replace(" ", "_").replace("-", "_")}.png')
                plt.savefig(grafico_path, dpi=300, bbox_inches='tight')
                graficos_paths.append((categoria, grafico_path))
                plt.close()
                
                print(f"  ✓ Gráfico generado: {categoria}")
            except Exception as e:
                print(f"  ✗ Error generando gráfico para {categoria}: {e}")
        
        # Crear PDF
        self.crear_pdf_reporte(graficos_paths, categorias_data, grafico_general)
        
        # Limpiar archivos temporales
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        return graficos_paths
    
    def crear_pdf_reporte(self, graficos_paths, categorias_data, grafico_general):
        """Crear PDF con todos los gráficos y análisis"""
        print("\nCreando PDF del reporte...")
        
        try:
            pdf_path = f'{OUTPUT_PATH}/Reporte [{datetime.now().strftime("%d-%m-%y • %H_%M")}].pdf'
            doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter), topMargin=0.5*inch, bottomMargin=0.5*inch)
            
            story = []
            styles = getSampleStyleSheet()
            
            # Título
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2E86AB'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            story.append(Paragraph("Reporte de Análisis de Precios - Machine Learning", title_style))
            
            # Información del reporte
            fecha_reporte = format_fecha_español(datetime.now())
            info_style = ParagraphStyle(
                'Info',
                parent=styles['Normal'],
                fontSize=10,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#666666')
            )
            story.append(Paragraph(f"Generado: {fecha_reporte}", info_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Resumen de datos
            summary_data = [
                ['Métrica', 'Valor'],
                ['Período de Datos', f"{self.data['Fecha'].min().strftime('%d/%m/%Y')} - {self.data['Fecha'].max().strftime('%d/%m/%Y')}"],
                ['Categorías Analizadas', f"{len(self.predictions)}"],
                ['Modelo de Predicción', 'Gradient Boosting Regressor'],
                ['Horizonte de Predicción', '6 meses'],
                ['Total de Registros', f"{len(self.data):,}"],
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')])
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Agregar gráfico general trimestral
            if grafico_general and os.path.exists(grafico_general):
                story.append(PageBreak())
                story.append(Paragraph("Evolución General de Precios", styles['Heading2']))
                story.append(Spacer(1, 0.2*inch))
                
                img_general = Image(grafico_general, width=9*inch, height=4.5*inch)
                story.append(img_general)
                story.append(Spacer(1, 0.2*inch))
                
                desc_general = Paragraph(
                    "<b>Análisis Trimestral:</b> Este gráfico muestra la evolución del precio promedio general "
                    "de todos los productos básicos durante el período 2008-2025, agrupado por trimestres. "
                    "El área sombreada representa el rango mínimo-máximo de variación en cada trimestre. "
                    "Esta visualización permite identificar tendencias generales, estacionalidad y cambios significativos en los precios.",
                    styles['Normal']
                )
                story.append(desc_general)
            
            # Agregar gráficos
            for categoria, grafico_path in sorted(graficos_paths):
                story.append(PageBreak())
                
                titulo_grafico = Paragraph(f"<b>{categoria}</b>", styles['Heading2'])
                story.append(titulo_grafico)
                story.append(Spacer(1, 0.2*inch))
                
                try:
                    if os.path.exists(grafico_path):
                        img = Image(grafico_path, width=9*inch, height=6*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                        
                        # Descripción
                        pred_data = self.predictions.get(categoria, {})
                        rmse = pred_data.get('rmse', 0)
                        desc = Paragraph(
                            f"<b>Análisis:</b> El gráfico superior muestra los precios de los últimos 30 días con su rango de variación. "
                            f"El gráfico inferior presenta el histórico de 6 meses junto con las predicciones del modelo Ridge Regression para "
                            f"el próximo período, incluyendo el intervalo de confianza del 95%.",
                            styles['Normal']
                        )
                        story.append(desc)
                except Exception as e:
                    print(f"  Error agregando imagen de {categoria}: {e}")
            
            # Conclusiones
            story.append(PageBreak())
            story.append(Paragraph("<b>Conclusiones y Metodología</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            # Formatear fechas en español
            fecha_inicio_esp = format_fecha_español(self.data['Fecha'].min().replace(hour=0, minute=0, second=0, microsecond=0))
            fecha_fin_esp = format_fecha_español(self.data['Fecha'].max().replace(hour=0, minute=0, second=0, microsecond=0))
            fecha_inicio_esp = fecha_inicio_esp.split(',')[0]
            fecha_fin_esp = fecha_fin_esp.split(',')[0]
            
            conclusiones = f"""
            <b>Periodo de Análisis:</b> {fecha_inicio_esp} - {fecha_fin_esp}<br/><br/>
            
            <b>Metodología:</b><br/>
            • Se consolidaron {len(self.data):,} registros de precios de productos básicos.<br/>
            • Los datos se agruparon por "Tipo de punto monitoreo" (categoría) y fecha, calculando el precio promedio.<br/>
            • Se entrenó un modelo Gradient Boosting Regressor con features avanzadas (lag, media móvil, tendencia, volatilidad).<br/>
            • Las predicciones se generaron para un horizonte de 6 meses con intervalo de confianza del 95%.<br/><br/>
            
            <b>Interpretación de Gráficos:</b><br/>
            • Gráfico Superior: Precios de los últimos 30 días con rango de variación (mínimo-máximo).<br/>
            • Gráfico Inferior: Histórico de 6 meses + predicciones del modelo Gradient Boosting. 
            La línea naranja muestra qué hubiera predicho el modelo en el pasado (validación), 
            la magenta son las predicciones futuras. La región sombreada es el intervalo de incertidumbre del 95%.<br/><br/>
            
            <b>Modelo Gradient Boosting Regressor:</b><br/>
            • Ensemble de árboles de decisión que aprenden secuencialmente<br/>
            • Cada árbol corrige los errores del árbol anterior<br/>
            • Utiliza features avanzadas: lags, media móvil, tendencia, volatilidad<br/>
            • Proporciona predicciones altamente precisas y confiables<br/>
            • Parámetros: 100 estimadores, profundidad máxima 5, learning rate 0.1<br/><br/>
            
            <b>Próximos Pasos Recomendados:</b><br/>
            • Actualizar regularmente los datos con información de ODEPA.<br/>
            • Re-entrenar los modelos mensualmente con nuevos datos.<br/>
            • Monitorear el RMSE para detectar cambios en los patrones de precios.<br/>
            • Considerar agregar variables externas (demanda, inflación, estacionalidad).<br/>
            """
            
            story.append(Paragraph(conclusiones, styles['Normal']))
            
            # Generar PDF
            doc.build(story)
            print(f"  ✓ PDF generado: {pdf_path}")
            
        except Exception as e:
            print(f"  ✗ Error creando PDF: {e}")
    
    def ejecutar_analisis_completo(self):
        """Ejecutar análisis completo"""
        print("=" * 60)
        print("ANÁLISIS DE PRECIOS CON MACHINE LEARNING")
        print("Modelo: Gradient Boosting Regressor")
        print("=" * 60)
        
        # 0. Descargar/actualizar CSV desde ODEPA
        self.descargar_csvs_odepa()
        
        # 1. Cargar datos
        self.cargar_datasets()
        
        # 2. Preparar datos por categoría
        categorias_data = self.preparar_datos_por_categoria()
        
        # 3. Entrenar modelos
        self.entrenar_modelos(categorias_data)
        
        # 4. Generar predicciones
        self.generar_predicciones(categorias_data)
        
        # 5. Generar gráficos y PDF
        self.generar_graficos_pdf(categorias_data)
        
        print("\n" + "=" * 60)
        print("ANÁLISIS COMPLETADO")
        print("=" * 60)
        print(f"\nResultados guardados en la carpeta: {OUTPUT_PATH}/")
        

def main():
    """Función principal"""
    try:
        analyzer = PreciosAnalyzer()
        analyzer.ejecutar_analisis_completo()
    except Exception as e:
        print(f"\nError crítico: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
