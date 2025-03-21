"""
Logging filters for secure data handling.
"""

import logging
import re
from typing import List, Optional, Dict, Any

class SensitiveDataFilter(logging.Filter):
    """Filter that masks sensitive data in log messages."""
    
    def __init__(self, patterns: Optional[List[str]] = None):
        """Initialize the filter with patterns to mask.
        
        Args:
            patterns: List of patterns to look for and mask in log messages
        """
        super().__init__()
        self.patterns = patterns or ['password', 'token', 'api_key', 'secret']
        self.compiled_patterns = [re.compile(f'({pattern})["\']?\\s*[=:]\\s*["\']?([^"\',\\s]+)', re.IGNORECASE) 
                                for pattern in self.patterns]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records by masking sensitive data.
        
        Args:
            record: The log record to filter
            
        Returns:
            bool: Always True (allows the record), but with modified content
        """
        if isinstance(record.msg, (str, bytes)):
            record.msg = self._mask_sensitive_data(str(record.msg))
        elif isinstance(record.msg, dict):
            record.msg = self._mask_sensitive_dict(record.msg)
        
        # Also check args for sensitive data
        if record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, (str, bytes)):
                    args[i] = self._mask_sensitive_data(str(arg))
                elif isinstance(arg, dict):
                    args[i] = self._mask_sensitive_dict(arg)
            record.args = tuple(args)
        
        return True

    def _mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text.
        
        Args:
            text: Text to mask sensitive data in
            
        Returns:
            str: Text with sensitive data masked
        """
        for pattern in self.compiled_patterns:
            text = pattern.sub('\\1=*****', text)
        return text

    def _mask_sensitive_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in dictionary.
        
        Args:
            data: Dictionary to mask sensitive data in
            
        Returns:
            Dict[str, Any]: Dictionary with sensitive data masked
        """
        masked_data = {}
        for key, value in data.items():
            # Mask values for sensitive keys
            if any(pattern in key.lower() for pattern in self.patterns):
                masked_data[key] = '*****'
            # Recursively mask nested dictionaries
            elif isinstance(value, dict):
                masked_data[key] = self._mask_sensitive_dict(value)
            # Mask sensitive data in strings
            elif isinstance(value, (str, bytes)):
                masked_data[key] = self._mask_sensitive_data(str(value))
            else:
                masked_data[key] = value
        return masked_data 