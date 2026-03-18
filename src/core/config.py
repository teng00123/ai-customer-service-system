import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用配置
    app_name: str = Field(default="AI智能客服系统", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # 服务器配置
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8080, env="PORT")
    
    # 数据库配置
    mongodb_uri: str = Field(default="mongodb://localhost:27017/customer_service", env="MONGODB_URI")
    mysql_uri: str = Field(default="mysql+pymysql://user:pass@localhost:3306/customer_service", env="MYSQL_URI")
    redis_uri: str = Field(default="redis://localhost:6379/0", env="REDIS_URI")
    elasticsearch_hosts: str = Field(default="http://localhost:9200", env="ELASTICSEARCH_HOSTS")
    
    # AI模型配置
    intent_model_path: str = Field(default="./models/intent_model", env="INTENT_MODEL_PATH")
    sentiment_model_path: str = Field(default="./models/sentiment_model", env="SENTIMENT_MODEL_PATH")
    similarity_threshold: float = Field(default=0.8, env="SIMILARITY_THRESHOLD")
    
    # JWT配置
    secret_key: str = Field(default="your-secret-key", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# 全局设置实例
settings = Settings()