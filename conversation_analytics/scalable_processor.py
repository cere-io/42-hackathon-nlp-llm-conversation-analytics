import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from datetime import datetime, timedelta
from .batch_processor import BatchProcessor
from .cache_manager import CacheManager
import faiss
from sklearn.cluster import DBSCAN
import ollama

class ScalableProcessor:
    def __init__(self, 
                 batch_size: int = 1000,
                 max_memory_mb: int = 1024,
                 cache_dir: str = "cache",
                 max_cache_age_days: int = 7,
                 model_name: str = "mistral"):
        """
        Inicializa el procesador escalable.
        
        Args:
            batch_size: Tamaño de los lotes de procesamiento
            max_memory_mb: Límite de memoria en MB
            cache_dir: Directorio para la caché
            max_cache_age_days: Edad máxima de la caché en días
            model_name: Nombre del modelo LLM a usar
        """
        self.batch_processor = BatchProcessor(batch_size, max_memory_mb)
        self.cache_manager = CacheManager(cache_dir, max_cache_age_days)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Inicializar índice FAISS para búsqueda de similitud
        self.vector_dimension = 384  # Dimensión del modelo all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.vector_dimension)
        
    def process_conversations(self, 
                            messages_df: pd.DataFrame,
                            model_config: Dict[str, Any],
                            force_reprocess: bool = False) -> pd.DataFrame:
        """
        Procesa conversaciones de manera escalable.
        
        Args:
            messages_df: DataFrame con los mensajes
            model_config: Configuración del modelo
            force_reprocess: Si se debe forzar el reprocesamiento
            
        Returns:
            DataFrame con los resultados procesados
        """
        # Generar clave de caché
        cache_key = self._generate_cache_key(messages_df, model_config)
        
        # Intentar recuperar de caché
        if not force_reprocess:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.info("Recuperando resultados de caché")
                return pd.DataFrame(cached_result)
        
        # Procesar en lotes
        results = []
        for batch in self.batch_processor.process_messages(messages_df):
            # Procesar lote
            batch_result = self._process_batch_with_model(batch, model_config)
            results.append(batch_result)
            
        # Combinar resultados
        final_result = pd.concat(results, ignore_index=True)
        
        # Guardar en caché
        self.cache_manager.set(cache_key, final_result.to_dict(orient='records'))
        
        return final_result
    
    def _process_batch_with_model(self, 
                                batch: pd.DataFrame,
                                model_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Procesa un lote con el modelo configurado.
        
        Args:
            batch: DataFrame con el lote
            model_config: Configuración del modelo
            
        Returns:
            DataFrame con los resultados del procesamiento
        """
        # Filtrar spam
        batch = batch[~batch['is_spam']]
        
        # Actualizar índice FAISS
        vectors = np.vstack(batch['vector'].tolist())
        self.index.add(vectors.astype('float32'))
        
        # Clustering de conversaciones
        conversation_ids = self._cluster_conversations(vectors)
        
        # Procesar con LLM
        topics = self._generate_topics_with_llm(batch, conversation_ids, model_config)
        
        return pd.DataFrame({
            'message_id': batch['message_id'],
            'conversation_id': conversation_ids,
            'topic': topics,
            'timestamp': batch['timestamp'],
            'confidence': self._calculate_confidence(batch, conversation_ids)
        })
    
    def _cluster_conversations(self, vectors: np.ndarray) -> List[int]:
        """
        Agrupa mensajes en conversaciones usando DBSCAN.
        
        Args:
            vectors: Vectores de los mensajes
            
        Returns:
            Lista de IDs de conversación
        """
        clustering = DBSCAN(
            eps=0.5,  # Distancia máxima entre muestras
            min_samples=2,  # Número mínimo de muestras en un cluster
            metric='euclidean'
        ).fit(vectors)
        
        return clustering.labels_.tolist()
    
    def _generate_topics_with_llm(self, 
                                batch: pd.DataFrame,
                                conversation_ids: List[int],
                                model_config: Dict[str, Any]) -> List[str]:
        """
        Genera temas para cada conversación usando el LLM.
        
        Args:
            batch: DataFrame con el lote
            conversation_ids: IDs de conversación
            model_config: Configuración del modelo
            
        Returns:
            Lista de temas
        """
        topics = []
        unique_conversations = set(conversation_ids)
        
        for conv_id in unique_conversations:
            if conv_id == -1:  # Mensajes sin cluster
                topics.extend(['Sin tema'] * sum(1 for cid in conversation_ids if cid == conv_id))
                continue
                
            # Obtener mensajes de la conversación
            conv_messages = batch[conversation_ids == conv_id]['text'].tolist()
            
            # Crear prompt para el LLM
            prompt = self._create_topic_prompt(conv_messages)
            
            try:
                # Generar tema con Ollama
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }]
                )
                
                topic = response['message']['content'].strip()
                topics.extend([topic] * len(conv_messages))
                
            except Exception as e:
                self.logger.error(f"Error al generar tema: {e}")
                topics.extend(['Error en generación'] * len(conv_messages))
        
        return topics
    
    def _create_topic_prompt(self, messages: List[str]) -> str:
        """
        Crea un prompt para el LLM.
        
        Args:
            messages: Lista de mensajes de la conversación
            
        Returns:
            Prompt formateado
        """
        return f"""Analiza los siguientes mensajes y genera un tema conciso y descriptivo que capture la esencia de la conversación:

Mensajes:
{chr(10).join(f"- {msg}" for msg in messages)}

Genera un tema que sea:
1. Específico y descriptivo
2. Capture el enfoque principal
3. Use términos técnicos cuando sea apropiado
4. Sea conciso (máximo 10 palabras)

Tema:"""
    
    def _calculate_confidence(self, 
                            batch: pd.DataFrame,
                            conversation_ids: List[int]) -> List[float]:
        """
        Calcula la confianza para cada mensaje.
        
        Args:
            batch: DataFrame con el lote
            conversation_ids: IDs de conversación
            
        Returns:
            Lista de valores de confianza
        """
        confidences = []
        for idx, conv_id in enumerate(conversation_ids):
            if conv_id == -1:
                confidences.append(0.0)
                continue
                
            # Calcular similitud con otros mensajes en la misma conversación
            conv_indices = [i for i, cid in enumerate(conversation_ids) if cid == conv_id]
            if len(conv_indices) == 1:
                confidences.append(0.5)
                continue
                
            # Calcular similitud promedio
            similarities = []
            for other_idx in conv_indices:
                if other_idx != idx:
                    sim = np.dot(batch.iloc[idx]['vector'], batch.iloc[other_idx]['vector'])
                    similarities.append(sim)
                    
            confidences.append(np.mean(similarities))
            
        return confidences
    
    def _generate_cache_key(self, 
                          messages_df: pd.DataFrame,
                          model_config: Dict[str, Any]) -> str:
        """
        Genera una clave única para la caché.
        
        Args:
            messages_df: DataFrame con los mensajes
            model_config: Configuración del modelo
            
        Returns:
            String con la clave única
        """
        # Crear un resumen de los datos para la clave
        data_summary = {
            'messages_count': len(messages_df),
            'date_range': {
                'start': messages_df['timestamp'].min().isoformat(),
                'end': messages_df['timestamp'].max().isoformat()
            },
            'model_config': model_config
        }
        
        return self.cache_manager._generate_cache_key(data_summary) 