# Conversation Detection System

## Descripción
Sistema avanzado para la detección y agrupación de conversaciones en mensajes de chat, utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático.

## Características Principales
- Detección automática de conversaciones usando múltiples técnicas
- Pre-agrupación basada en tiempo, semántica y patrones de usuario
- Evaluación exhaustiva con múltiples métricas
- Visualizaciones detalladas de resultados
- Soporte para diferentes modelos de lenguaje

## Requisitos
- Python 3.8+
- Dependencias listadas en `requirements.txt`
- GPU opcional (recomendado para modelos grandes)

## Instalación
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Estructura del Proyecto
```
.
├── src/
│   ├── metrics/          # Módulos de evaluación
│   ├── models/          # Modelos de ML
│   └── utils/           # Utilidades comunes
├── optimization/        # Scripts de optimización
├── evaluation/         # Scripts de evaluación
├── tests/             # Tests unitarios
├── data/              # Datos de entrada/salida
└── results/           # Resultados y visualizaciones
```

## Uso

### Pre-agrupación de Mensajes
```bash
python optimization/pre_grouping_techniques.py data/groups/thisiscere/messages_thisiscere.csv
```

### Evaluación de Resultados
```bash
python src/metrics/conversation_metrics.py data/groups/thisiscere
```

### Optimización de Modelos
```bash
python optimization/experiment_models.py data/groups/thisiscere/messages_thisiscere.csv
```

## Métricas y Evaluación
El sistema utiliza múltiples métricas para evaluar la calidad de las agrupaciones:
- Adjusted Rand Index (ARI)
- Coherencia temporal
- Coherencia semántica
- Estadísticas de grupos

## Visualizaciones
- Gráficos de puntuación ARI a lo largo del tiempo
- Métricas de coherencia
- Mapas de calor de correlación
- Visualizaciones de grupos temporales

## Contribución
1. Fork el repositorio
2. Cree una rama para su feature (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abra un Pull Request

## Licencia
Este proyecto está licenciado bajo la Licencia MIT - vea el archivo `LICENSE` para más detalles.

## Contacto
Nombre - [@twitter_handle](https://twitter.com/twitter_handle)
Email - email@example.com

## Agradecimientos
- Agradecimiento especial a todos los contribuidores
- Inspirado en las mejores prácticas de NLP y análisis de conversaciones
- Basado en investigaciones recientes en el campo de procesamiento de lenguaje natural 