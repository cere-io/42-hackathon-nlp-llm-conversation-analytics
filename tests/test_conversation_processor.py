import pandas as pd
import logging
from pathlib import Path
from conversation_analytics.scalable_processor import ScalableProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_processor():
    try:
        # Crear directorios necesarios
        Path("cache").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
        # Cargar datos de prueba
        logger.info("Cargando datos de prueba...")
        messages_df = pd.read_csv("data/groups/thisiscere/messages_thisiscere.csv")
        logger.info(f"Datos cargados: {len(messages_df)} mensajes")
        
        # Configuración del modelo
        model_config = {
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Inicializar y ejecutar el procesador
        logger.info("Iniciando procesamiento...")
        processor = ScalableProcessor(
            batch_size=100,  # Tamaño de lote más pequeño para pruebas
            max_memory_mb=2048,
            cache_dir="cache",
            max_cache_age_days=7,
            model_name="mistral"
        )
        
        # Procesar conversaciones
        results = processor.process_conversations(
            messages_df=messages_df,
            model_config=model_config,
            force_reprocess=True  # Forzar reprocesamiento para pruebas
        )
        
        # Guardar resultados
        output_path = "results/processed_conversations.csv"
        results.to_csv(output_path, index=False)
        logger.info(f"Resultados guardados en {output_path}")
        
        # Mostrar estadísticas
        logger.info("\nEstadísticas de procesamiento:")
        logger.info(f"Total de mensajes procesados: {len(results)}")
        logger.info(f"Total de conversaciones únicas: {results['conversation_id'].nunique()}")
        logger.info(f"Promedio de confianza: {results['confidence'].mean():.3f}")
        
        # Mostrar algunas conversaciones de ejemplo
        logger.info("\nEjemplos de conversaciones procesadas:")
        for conv_id in results['conversation_id'].unique()[:3]:
            conv_messages = results[results['conversation_id'] == conv_id]
            logger.info(f"\nConversación {conv_id}:")
            logger.info(f"Tema: {conv_messages['topic'].iloc[0]}")
            logger.info(f"Confianza promedio: {conv_messages['confidence'].mean():.3f}")
            logger.info(f"Número de mensajes: {len(conv_messages)}")
            
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {e}")
        raise

if __name__ == "__main__":
    test_processor() 