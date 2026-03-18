# AI智能客服系统

基于自然语言处理和机器学习的智能客服解决方案，提供24/7全天候客户服务。

## 项目特性

- 🤖 多轮对话理解和上下文保持
- 🎯 高精度意图识别和实体抽取  
- 📚 智能知识库检索和相似度匹配
- 💡 个性化推荐和情感分析
- ⚡ 3秒内快速响应
- 📈 85%+自助解决率

## 技术架构

- **前端**: Web Chat, Mobile App, 第三方集成
- **后端**: 微服务架构，基于FastAPI
- **AI模型**: BERT/GPT意图识别，LSTM情感分析，TensorFlow推荐
- **数据库**: MongoDB(会话), MySQL(用户), Elasticsearch(知识库), Redis(缓存)
- **部署**: Docker容器化，Kubernetes编排

## 快速开始

### 环境要求

- Python 3.8+
- Docker & Docker Compose
- Redis
- MongoDB
- MySQL 8.0+
- Elasticsearch 7.x

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/teng00123/ai-customer-service-system.git
cd ai-customer-service-system
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
```bash
cp config/env.example .env
# 编辑.env文件配置数据库连接等信息
```

4. 启动服务
```bash
docker-compose up -d
```

5. 运行数据库迁移
```bash
python scripts/migrate.py
```

## 项目结构

```
ai-customer-service-system/
├── src/                    # 源代码
│   ├── api/               # API接口层
│   ├── core/              # 核心业务逻辑
│   ├── intent_recognition/ # 意图识别引擎
│   ├── knowledge_base/    # 知识库系统
│   ├── recommendation_engine/ # 推荐引擎
│   └── dialog_manager/    # 对话管理器
├── config/                # 配置文件
│   ├── docker/            # Docker配置
│   └── nginx/             # Nginx配置
├── tests/                 # 测试文件
├── scripts/               # 部署脚本
├── data/                  # 数据文件
│   └── migrations/        # 数据库迁移
├── docs/                  # 文档
└── requirements.txt       # Python依赖
```

## API文档

启动服务后访问: http://localhost:8080/docs

### 核心接口

- `POST /api/v1/chat/message` - 发送消息
- `GET /api/v1/chat/response/{session_id}` - 获取回复
- `GET /api/v1/conversations/{session_id}` - 获取对话历史

## 开发指南

### 添加新意图

1. 在 `src/intent_recognition/data/intents.json` 中添加意图定义
2. 准备训练数据到 `data/training/intents/`
3. 重新训练模型: `python src/intent_recognition/train.py`

### 更新知识库

1. 通过管理界面或直接操作Elasticsearch
2. 支持批量导入CSV/JSON格式

## 部署

### Docker部署

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes部署

```bash
kubectl apply -f k8s/
```

## 监控

- Prometheus metrics: http://localhost:9090
- Grafana dashboards: http://localhost:3000
- 日志聚合: ELK Stack

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目主页: https://github.com/teng00123/ai-customer-service-system
- 问题反馈: https://github.com/teng00123/ai-customer-service-system/issues
- 邮箱: your-email@example.com

---

**版本**: v1.0.0  
**更新时间**: 2026-03-18