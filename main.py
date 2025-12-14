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

# Machine Learning
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
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

# Configuración
DATASETS_PATH = 'datasets'
OUTPUT_PATH = 'output'
MODELS_PATH = 'models'

# Crear directorios si no existen
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

class PreciosAnalyzer:
    """Clase para analizar precios de productos básicos"""
    
    def __init__(self):
        self.data = None
        self.data_odepa = None
        self.models = {}
        self.predictions = {}
        
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
    
    def obtener_datos_odepa(self):
        """Intentar obtener datos nuevos desde ODEPA"""
        print("\nIntentando obtener datos de ODEPA...")
        try:
            # Usar API de ODEPA
            url = "https://datos.odepa.gob.cl/api/3/action/datastore_search"
            params = {
                'resource_id': '95e92da3-e1f8-404f-af11-4e3ba8f29de9',  # ID del dataset de precios
                'limit': 10000
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data_json = response.json()
                if 'result' in data_json and 'records' in data_json['result']:
                    records = data_json['result']['records']
                    self.data_odepa = pd.DataFrame(records)
                    print(f"  ✓ Datos ODEPA obtenidos: {len(self.data_odepa)} registros")
                    return True
        except Exception as e:
            print(f"  ✗ No se pudo obtener datos de ODEPA: {e}")
        
        print("  Continuando con datos históricos disponibles...")
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
        """Entrenar modelos ARIMA para cada categoría"""
        print("\nEntrenando modelos ML...")
        
        for categoria, datos in categorias_data.items():
            try:
                # Usar ARIMA para series de tiempo
                serie = datos['Precio promedio'].values
                
                # Solo entrenar si hay suficientes datos
                if len(serie) > 20:
                    try:
                        # Probar ARIMA(1,1,1) como configuración inicial
                        modelo = ARIMA(serie, order=(1, 1, 1))
                        modelo_fit = modelo.fit()
                        self.models[categoria] = modelo_fit
                        print(f"  ✓ {categoria}: ARIMA(1,1,1) - AIC: {modelo_fit.aic:.2f}")
                    except:
                        # Si falla, usar configuración más simple
                        modelo = ARIMA(serie, order=(1, 0, 0))
                        modelo_fit = modelo.fit()
                        self.models[categoria] = modelo_fit
                        print(f"  ✓ {categoria}: ARIMA(1,0,0) - AIC: {modelo_fit.aic:.2f}")
                else:
                    print(f"  ⚠ {categoria}: Datos insuficientes para entrenar")
            except Exception as e:
                print(f"  ✗ Error en {categoria}: {e}")
    
    def generar_predicciones(self, categorias_data, meses_adelante=6):
        """Generar predicciones para los próximos meses"""
        print(f"\nGenerando predicciones para {meses_adelante} meses...")
        
        predicciones = {}
        
        for categoria, modelo_fit in self.models.items():
            try:
                datos = categorias_data[categoria]
                fecha_ultima = datos['Fecha'].max()
                
                # Predicciones
                forecast = modelo_fit.get_forecast(steps=meses_adelante * 4)  # Aproximadamente 4 semanas por mes
                pred_mean = forecast.predicted_mean
                pred_ci = forecast.conf_int()
                
                # Convertir a arrays
                if isinstance(pred_mean, pd.Series):
                    pred_mean_values = pred_mean.values
                else:
                    pred_mean_values = np.array(pred_mean)
                
                if isinstance(pred_ci, pd.DataFrame):
                    lower_ci_values = pred_ci.iloc[:, 0].values
                    upper_ci_values = pred_ci.iloc[:, 1].values
                else:
                    lower_ci_values = np.array(pred_ci[:, 0])
                    upper_ci_values = np.array(pred_ci[:, 1])
                
                # Crear fechas futuras
                fechas_futuras = pd.date_range(start=fecha_ultima + timedelta(days=7), periods=len(pred_mean_values), freq='7D')
                
                predicciones[categoria] = {
                    'fechas': fechas_futuras,
                    'predicciones': pred_mean_values,
                    'lower_ci': lower_ci_values,
                    'upper_ci': upper_ci_values,
                    'datos_historicos': datos
                }
                
                print(f"  ✓ {categoria}: {len(pred_mean_values)} predicciones")
            except Exception as e:
                print(f"  ✗ Error prediciendo {categoria}: {e}")
        
        self.predictions = predicciones
        return predicciones
    
    def calcular_ultimos_30_dias(self, categorias_data):
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
        print("\nGenerando gráficos y PDF...")
        
        # Configurar matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        graficos_paths = []
        
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
                ax1.xaxis.set_major_formatter(DateFormatter('%d-%m'))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                
                # Gráfico 2: Predicciones vs Histórico (6 meses)
                ax2 = axes[1]
                
                # Datos históricos (6 meses)
                fecha_inicio_6m = pred_data['fechas'][0] - timedelta(days=180)
                datos_6m = pred_data['datos_historicos'][
                    pred_data['datos_historicos']['Fecha'] >= fecha_inicio_6m
                ]
                
                ax2.plot(datos_6m['Fecha'], datos_6m['Precio promedio'], 
                        marker='o', linestyle='-', linewidth=2, label='Histórico', color='#2E86AB')
                
                # Predicciones
                ax2.plot(pred_data['fechas'], pred_data['predicciones'], 
                        marker='s', linestyle='--', linewidth=2, label='Predicción ML', color='#A23B72')
                
                # Intervalo de confianza
                ax2.fill_between(pred_data['fechas'], 
                                pred_data['lower_ci'], 
                                pred_data['upper_ci'], 
                                alpha=0.2, color='#A23B72', label='Intervalo 95%')
                
                ax2.set_title(f'Proyección 6 Meses y Predicciones - {categoria}', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Precio ($)', fontsize=11)
                ax2.set_xlabel('Fecha', fontsize=11)
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)
                ax2.xaxis.set_major_formatter(DateFormatter('%d-%m'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                
                plt.tight_layout()
                
                # Guardar gráfico
                grafico_path = f'{OUTPUT_PATH}/grafico_{categoria.replace(" ", "_").replace("-", "_")}.png'
                plt.savefig(grafico_path, dpi=300, bbox_inches='tight')
                graficos_paths.append((categoria, grafico_path))
                plt.close()
                
                print(f"  ✓ Gráfico generado: {categoria}")
            except Exception as e:
                print(f"  ✗ Error generando gráfico para {categoria}: {e}")
        
        # Crear PDF
        self.crear_pdf_reporte(graficos_paths, categorias_data)
        
        return graficos_paths
    
    def crear_pdf_reporte(self, graficos_paths, categorias_data):
        """Crear PDF con todos los gráficos y análisis"""
        print("\nCreando PDF del reporte...")
        
        try:
            pdf_path = f'{OUTPUT_PATH}/Reporte [{datetime.now().strftime("%d-%m-%y")}].pdf'
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
            fecha_reporte = datetime.now().strftime("%d de %B de %Y, %H:%M")
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
                ['Modelo de Predicción', 'ARIMA (AutoRegressive Integrated Moving Average)'],
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
                        desc = Paragraph(
                            f"<b>Análisis:</b> El gráfico superior muestra los precios de los últimos 30 días con su rango de variación. "
                            f"El gráfico inferior presenta el histórico de 6 meses junto con las predicciones del modelo ARIMA para "
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
            
            conclusiones = f"""
            <b>Periodo de Análisis:</b> {self.data['Fecha'].min().strftime('%d de %B de %Y')} - {self.data['Fecha'].max().strftime('%d de %B de %Y')}<br/><br/>
            
            <b>Metodología:</b><br/>
            • Se consolidaron {len(self.data):,} registros de precios de productos básicos.<br/>
            • Los datos se agruparon por "Tipo de punto monitoreo" (categoría) y fecha, calculando el precio promedio.<br/>
            • Se entrenó un modelo ARIMA (AutoRegressive Integrated Moving Average) para cada categoría.<br/>
            • Las predicciones se generaron para un horizonte de 6 meses con intervalo de confianza del 95%.<br/><br/>
            
            <b>Interpretación de Gráficos:</b><br/>
            • Gráfico Superior: Muestra los precios de los últimos 30 días. El área sombreada representa el rango entre precio mínimo y máximo.<br/>
            • Gráfico Inferior: Compara los precios históricos (últimos 6 meses) con las predicciones del modelo. 
            La región sombreada alrededor de la línea de predicción representa el intervalo de incertidumbre.<br/><br/>
            
            <b>Modelo ARIMA:</b><br/>
            El modelo ARIMA es especialmente efectivo para series de tiempo que muestran patrones de autocorrelación 
            y tendencias. Captura tres componentes principales:<br/>
            • AR (AutoRegressive): Dependencia de valores anteriores<br/>
            • I (Integrated): Diferenciación para hacerla estacionaria<br/>
            • MA (Moving Average): Dependencia de errores anteriores<br/><br/>
            
            <b>Próximos Pasos Recomendados:</b><br/>
            • Actualizar regularmente los datos con información de ODEPA.<br/>
            • Re-entrenar los modelos mensualmente con nuevos datos.<br/>
            • Considerar factores exógenos (inflación, tipos de cambio, estacionalidad).<br/>
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
        print("=" * 60)
        
        # 1. Cargar datos
        self.cargar_datasets()
        
        # 2. Intentar obtener datos nuevos de ODEPA
        self.obtener_datos_odepa()
        
        # 3. Preparar datos por categoría
        categorias_data = self.preparar_datos_por_categoria()
        
        # 4. Entrenar modelos
        self.entrenar_modelos(categorias_data)
        
        # 5. Generar predicciones
        self.generar_predicciones(categorias_data)
        
        # 6. Generar gráficos y PDF
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
