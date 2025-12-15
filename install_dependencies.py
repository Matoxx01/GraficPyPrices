"""
Script para instalar las dependencias necesarias para el análisis de precios
"""

import subprocess
import sys

def instalar_dependencias():
    """Instalar paquetes requeridos"""
    
    paquetes = [
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.2.0',
        'statsmodels>=0.13.0',
        'pmdarima>=2.0.0',  # Para Auto-ARIMA
        'joblib>=1.2.0',
        'requests>=2.28.0',
        'reportlab>=4.0.0',
        'Pillow>=9.0.0',
        'python-dateutil>=2.8.0',
    ]
    
    print("Instalando dependencias necesarias...")
    print("=" * 60)
    
    for paquete in paquetes:
        print(f"Instalando {paquete}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", paquete])
            print(f"  ✓ {paquete} instalado")
        except subprocess.CalledProcessError:
            print(f"  ✗ Error instalando {paquete}")
    
    print("=" * 60)
    print("Instalación completada")

if __name__ == "__main__":
    instalar_dependencias()
