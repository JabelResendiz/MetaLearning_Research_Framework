#!/bin/bash

# Script para ejecutar el proyecto MetaLearning
# Facilita la configuración del entorno y ejecución del proyecto

set -e  # Salir si hay algún error

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MetaLearning Project Setup & Run${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: Python 3 no está instalado${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓${NC} Python encontrado: $PYTHON_VERSION"

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creando entorno virtual...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Entorno virtual creado"
else
    echo -e "${GREEN}✓${NC} Entorno virtual ya existe"
fi

# Activar entorno virtual
echo -e "${YELLOW}Activando entorno virtual...${NC}"
source venv/bin/activate

# Actualizar pip
echo -e "${YELLOW}Actualizando pip...${NC}"
pip install --upgrade pip --quiet

# Instalar dependencias
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Instalando dependencias...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Dependencias instaladas"
else
    echo -e "${YELLOW}Advertencia: requirements.txt no encontrado${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓${NC} Configuración completada"
echo -e "${BLUE}========================================${NC}"
echo ""

# Menú de opciones
echo "¿Qué deseas hacer?"
echo "1) Abrir Jupyter Notebook"
echo "2) Ejecutar ejemplo de carga de datos"
echo "3) Solo activar el entorno virtual"
echo "4) Salir"
echo ""
read -p "Selecciona una opción (1-4): " option

case $option in
    1)
        echo -e "${YELLOW}Iniciando Jupyter Notebook...${NC}"
        echo -e "${GREEN}El notebook se abrirá en tu navegador${NC}"
        jupyter notebook notebooks/
        ;;
    2)
        echo -e "${YELLOW}Ejecutando ejemplo de carga de datos...${NC}"
        python3 -c "
import sys
sys.path.append('src')
from data_loader import load_openml_dataset
print('Cargando dataset Iris (ID: 61)...')
dataset = load_openml_dataset(61)
if dataset:
    print(f'✓ Dataset cargado: {dataset[\"name\"]}')
    print(f'  - Muestras: {dataset[\"metadata\"][\"n_samples\"]}')
    print(f'  - Características: {dataset[\"metadata\"][\"n_features\"]}')
    print(f'  - Clases: {dataset[\"metadata\"][\"n_classes\"]}')
else:
    print('✗ Error al cargar el dataset')
"
        ;;
    3)
        echo -e "${GREEN}Entorno virtual activado${NC}"
        echo -e "${YELLOW}Para desactivar, ejecuta: deactivate${NC}"
        echo ""
        echo -e "${BLUE}Puedes ejecutar comandos Python ahora...${NC}"
        exec $SHELL
        ;;
    4)
        echo -e "${GREEN}¡Hasta luego!${NC}"
        deactivate 2>/dev/null || true
        exit 0
        ;;
    *)
        echo -e "${YELLOW}Opción no válida${NC}"
        deactivate 2>/dev/null || true
        exit 1
        ;;
esac

