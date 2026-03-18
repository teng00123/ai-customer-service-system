"""模型管理器 - Claude Code风格"""
import os
import shutil
import hashlib
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

class ModelManager:
    """AI模型管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.model_cache = {}  # 模型缓存
        self.model_registry = {}  # 模型注册表
        self.model_storage_path = Path(self.config.get('storage_path', 'models'))
        self.max_cache_size = self.config.get('cache_size', 3)
        
        # 创建存储目录
        self.model_storage_path.mkdir(exist_ok=True)
        
        logger.info(f"Model manager initialized with storage: {self.model_storage_path}")
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'storage_path': 'models',
            'cache_size': 3,
            'auto_cleanup': True,
            'cleanup_interval_days': 7,
            'model_validation': True,
            'backup_enabled': True
        }
    
    async def load_model(self, model_id: str, model_type: str, 
                       model_path: str, **kwargs) -> Any:
        """加载模型"""
        try:
            # 检查缓存
            cache_key = f"{model_id}_{model_type}"
            if cache_key in self.model_cache:
                logger.info(f"Loading model from cache: {model_id}")
                return self.model_cache[cache_key]
            
            # 验证模型文件
            if self.config.get('model_validation', True):
                if not await self._validate_model_file(model_path):
                    raise ValueError(f"Invalid model file: {model_path}")
            
            # 加载模型
            if model_type == 'pytorch':
                model = await self._load_pytorch_model(model_path, **kwargs)
            elif model_type == 'tensorflow':
                model = await self._load_tensorflow_model(model_path, **kwargs)
            elif model_type == 'sklearn':
                model = await self._load_sklearn_model(model_path, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # 更新缓存
            self._update_cache(cache_key, model)
            
            # 注册模型
            self.model_registry[model_id] = {
                'model_type': model_type,
                'model_path': model_path,
                'loaded_at': datetime.now(),
                'last_used': datetime.now(),
                'usage_count': 1
            }
            
            logger.info(f"Model loaded successfully: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    async def _validate_model_file(self, model_path: str) -> bool:
        """验证模型文件完整性"""
        try:
            if not os.path.exists(model_path):
                return False
                
            # 检查文件大小
            file_size = os.path.getsize(model_path)
            if file_size == 0:
                return False
                
            # 检查文件哈希（可选）
            if self.config.get('checksum_validation', False):
                expected_hash = self.config.get('model_checksums', {}).get(model_path)
                if expected_hash:
                    actual_hash = self._calculate_file_hash(model_path)
                    if actual_hash != expected_hash:
                        logger.warning(f"Model checksum mismatch: {model_path}")
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def _load_pytorch_model(self, model_path: str, **kwargs) -> torch.nn.Module:
        """加载PyTorch模型"""
        try:
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, dict) and 'model_state_dict' in model:
                # 加载模型结构
                model_class = kwargs.get('model_class')
                if model_class:
                    model_instance = model_class(**kwargs.get('model_params', {}))
                    model_instance.load_state_dict(model['model_state_dict'])
                    return model_instance
                else:
                    logger.warning("No model class provided, returning state dict")
                    return model
            return model
            
        except Exception as e:
            logger.error(f"PyTorch model loading failed: {e}")
            raise
    
    async def _load_tensorflow_model(self, model_path: str, **kwargs) -> Any:
        """加载TensorFlow模型"""
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        except ImportError:
            logger.error("TensorFlow not available")
            raise
        except Exception as e:
            logger.error(f"TensorFlow model loading failed: {e}")
            raise
    
    async def _load_sklearn_model(self, model_path: str, **kwargs) -> Any:
        """加载Scikit-learn模型"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Sklearn model loading failed: {e}")
            raise
    
    def _update_cache(self, cache_key: str, model: Any) -> None:
        """更新模型缓存"""
        # 如果缓存已满，移除最久未使用的模型
        if len(self.model_cache) >= self.max_cache_size:
            self._evict_cache()
            
        self.model_cache[cache_key] = {
            'model': model,
            'cached_at': datetime.now()
        }
    
    def _evict_cache(self) -> None:
        """缓存淘汰策略 (LRU)"""
        if not self.model_cache:
            return
            
        # 找到最久未使用的模型
        oldest_key = min(
            self.model_cache.keys(),
            key=lambda k: self.model_cache[k]['cached_at']
        )
        
        del self.model_cache[oldest_key]
        logger.info(f"Evicted model from cache: {oldest_key}")
    
    async def save_model(self, model_id: str, model: Any, model_type: str, 
                       metadata: Dict[str, Any] = None) -> str:
        """保存模型"""
        try:
            # 生成文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_id}_{timestamp}.pth"
            model_path = self.model_storage_path / filename
            
            # 备份原模型（如果存在）
            if self.config.get('backup_enabled', True):
                await self._backup_existing_model(model_id)
            
            # 保存模型
            if model_type == 'pytorch':
                torch.save(model, str(model_path))
            elif model_type == 'sklearn':
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            else:
                raise ValueError(f"Unsupported save format for type: {model_type}")
            
            # 保存元数据
            if metadata:
                meta_path = model_path.with_suffix('.meta')
                with open(meta_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise
    
    async def _backup_existing_model(self, model_id: str) -> None:
        """备份现有模型"""
        try:
            # 查找该模型的现有文件
            existing_files = list(self.model_storage_path.glob(f"{model_id}_*.pth"))
            for file_path in existing_files[:-1]:  # 保留最新的一个
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backed up model: {file_path} -> {backup_path}")
        except Exception as e:
            logger.warning(f"Model backup failed: {e}")
    
    async def cleanup_old_models(self) -> None:
        """清理旧模型文件"""
        if not self.config.get('auto_cleanup', True):
            return
            
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.get('cleanup_interval_days', 7))
            cleaned_count = 0
            
            for model_file in self.model_storage_path.glob("*.pth"):
                if model_file.stat().st_mtime < cutoff_date.timestamp():
                    model_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Cleaned old model: {model_file}")
                    
            logger.info(f"Model cleanup completed: removed {cleaned_count} files")
            
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        return self.model_registry.get(model_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型"""
        return [
            {
                'model_id': model_id,
                **info
            }
            for model_id, info in self.model_registry.items()
        ]
    
    async def unload_model(self, model_id: str) -> bool:
        """卸载模型"""
        try:
            # 从缓存中移除
            keys_to_remove = [k for k in self.model_cache.keys() if k.startswith(f"{model_id}_")]
            for key in keys_to_remove:
                del self.model_cache[key]
                
            # 从注册表中移除
            if model_id in self.model_registry:
                del self.model_registry[model_id]
                
            logger.info(f"Model unloaded: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'cache_size': len(self.model_cache),
            'max_cache_size': self.max_cache_size,
            'cached_models': list(self.model_cache.keys()),
            'registry_size': len(self.model_registry)
        }