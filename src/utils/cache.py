"""缓存管理器 - Claude Code风格"""
import logging
import pickle
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)

class CacheManager:
    """简单的缓存管理器实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ttl = config.get('ttl', 3600)  # 默认1小时
        self.max_size = config.get('max_size', 1000)
        self._cache = {}  # 内存缓存
        
        logger.info(f"Cache manager initialized with TTL={self.ttl}s, max_size={self.max_size}")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            if key in self._cache:
                cached_item = self._cache[key]
                
                # 检查是否过期
                if datetime.now() < cached_item['expires_at']:
                    logger.debug(f"Cache hit for key: {key}")
                    return cached_item['value']
                else:
                    # 过期了，删除
                    del self._cache[key]
                    logger.debug(f"Cache expired for key: {key}")
            
            logger.debug(f"Cache miss for key: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            # 检查缓存大小限制
            if len(self._cache) >= self.max_size:
                # 简单的LRU清理：删除最旧的条目
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['created_at'])
                del self._cache[oldest_key]
                logger.debug(f"Cache size limit reached, removed oldest key: {oldest_key}")
            
            # 设置缓存项
            expires_at = datetime.now() + timedelta(seconds=ttl or self.ttl)
            self._cache[key] = {
                'value': value,
                'created_at': datetime.now(),
                'expires_at': expires_at
            }
            
            logger.debug(f"Cache set for key: {key}, expires at: {expires_at}")
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        try:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache deleted for key: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的所有缓存项"""
        try:
            import fnmatch
            keys_to_delete = [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
            
            for key in keys_to_delete:
                del self._cache[key]
            
            logger.debug(f"Cleared {len(keys_to_delete)} cache entries matching pattern: {pattern}")
            return len(keys_to_delete)
            
        except Exception as e:
            logger.error(f"Cache clear pattern failed for pattern {pattern}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        now = datetime.now()
        active_entries = sum(1 for item in self._cache.values() if now < item['expires_at'])
        expired_entries = len(self._cache) - active_entries
        
        return {
            'total_entries': len(self._cache),
            'active_entries': active_entries,
            'expired_entries': expired_entries,
            'cache_size_mb': len(str(self._cache)) / (1024 * 1024),
            'ttl_seconds': self.ttl,
            'max_size': self.max_size
        }
    
    def clear_all(self) -> None:
        """清空所有缓存"""
        self._cache.clear()
        logger.info("All cache entries cleared")