import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from datetime import datetime, timedelta

class CacheManager:
    def __init__(self, cache_dir: str = "cache", max_age_days: int = 7):
        """
        Inicializa el gestor de caché.
        
        Args:
            cache_dir: Directorio para almacenar la caché
            max_age_days: Edad máxima de los archivos en caché en días
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self.logger = logging.getLogger(__name__)
        
    def _generate_cache_key(self, data: Any) -> str:
        """
        Genera una clave única para los datos.
        
        Args:
            data: Datos a cachear
            
        Returns:
            String con la clave única
        """
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
            
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Recupera datos de la caché.
        
        Args:
            key: Clave de los datos a recuperar
            
        Returns:
            Datos cacheados o None si no existen o están expirados
        """
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Verificar expiración
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(days=self.max_age_days):
                self.logger.info(f"Cache expirada para clave {key}")
                return None
                
            return cache_data['value']
            
        except Exception as e:
            self.logger.error(f"Error al leer caché: {e}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        Almacena datos en la caché.
        
        Args:
            key: Clave para los datos
            value: Datos a almacenar
            
        Returns:
            True si se almacenó correctamente, False en caso contrario
        """
        try:
            cache_data = {
                'value': value,
                'timestamp': datetime.now().isoformat()
            }
            
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error al escribir en caché: {e}")
            return False
    
    def clear_expired(self) -> int:
        """
        Elimina archivos de caché expirados.
        
        Returns:
            Número de archivos eliminados
        """
        count = 0
        now = datetime.now()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if now - cache_time > timedelta(days=self.max_age_days):
                    cache_file.unlink()
                    count += 1
                    
            except Exception as e:
                self.logger.error(f"Error al procesar archivo de caché {cache_file}: {e}")
                
        return count 