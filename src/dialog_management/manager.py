"""对话管理器 - Claude Code风格"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid

from .session import DialogSession
from .context import DialogContext
from .state_machine import DialogStateMachine
from ..ai_engine.engine import AIModelEngine
from ..knowledge_base.manager import KnowledgeBaseManager
from ..intent_recognition.models import ModelLoader

logger = logging.getLogger(__name__)

class DialogManager:
    """对话管理器主控制器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.sessions = {}  # session_id -> DialogSession
        self.active_sessions = defaultdict(set)  # user_id -> set(session_id)
        self.contexts = {}  # session_id -> DialogContext
        self.state_machines = {}  # session_id -> DialogStateMachine
        
        # AI引擎集成
        self.ai_engine = AIModelEngine(self.config.get('ai_engine', {}))
        self.kb_manager = KnowledgeBaseManager(self.config.get('knowledge_base', {}))
        self.model_loader = ModelLoader()
        
        # 会话清理配置
        self.session_timeout = timedelta(hours=self.config.get('session_timeout_hours', 24))
        self.max_sessions_per_user = self.config.get('max_sessions_per_user', 5)
        
        # 统计信息
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_messages': 0,
            'average_response_time': 0.0,
            'last_updated': datetime.now()
        }
        
        logger.info("Dialog manager initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'session_timeout_hours': 24,
            'max_sessions_per_user': 5,
            'enable_context_memory': True,
            'enable_knowledge_retrieval': True,
            'enable_emotion_detection': True,
            'auto_session_cleanup': True,
            'ai_engine': {
                'nlp': {'enable_spell_check': True},
                'recommendation': {'enabled': True}
            },
            'knowledge_base': {
                'search': {'max_results': 3}
            }
        }
    
    async def initialize(self) -> None:
        """初始化对话管理器"""
        try:
            # 初始化AI引擎
            await self.ai_engine.initialize()
            
            # 启动清理任务
            if self.config.get('auto_session_cleanup', True):
                asyncio.create_task(self._cleanup_expired_sessions())
                
            logger.info("Dialog manager initialization completed")
            
        except Exception as e:
            logger.error(f"Dialog manager initialization failed: {e}")
            raise
    
    async def create_dialog(self, user_id: str, metadata: Dict[str, Any] = None) -> DialogSession:
        """创建新对话"""
        try:
            # 检查用户会话数量限制
            user_sessions = self.active_sessions[user_id]
            if len(user_sessions) >= self.max_sessions_per_user:
                # 关闭最旧的会话
                oldest_session_id = min(
                    user_sessions,
                    key=lambda sid: self.sessions[sid].created_at
                )
                await self.close_dialog(oldest_session_id)
            
            # 创建会话
            session = DialogSession(
                user_id=user_id,
                metadata=metadata or {}
            )
            
            # 创建上下文
            context = DialogContext(session_id=session.id, user_id=user_id)
            
            # 创建状态机
            state_machine = DialogStateMachine(initial_state="greeting")
            
            # 存储组件
            self.sessions[session.id] = session
            self.contexts[session.id] = context
            self.state_machines[session.id] = state_machine
            self.active_sessions[user_id].add(session.id)
            
            # 更新统计
            self.stats['total_sessions'] += 1
            self.stats['active_sessions'] = len(self.sessions)
            
            logger.info(f"Created dialog session: {session.id} for user: {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create dialog: {e}")
            raise
    
    async def process_message(self, session_id: str, user_id: str, 
                           message: str, message_type: str = "text") -> Dict[str, Any]:
        """处理用户消息"""
        start_time = datetime.now()
        
        try:
            # 获取或创建会话
            session = await self._get_or_create_session(session_id, user_id)
            context = self.contexts[session.id]
            state_machine = self.state_machines[session.id]
            
            # 更新会话活动
            session.update_activity()
            context.update_context(user_id, message, message_type)
            
            # 情感检测
            if self.config.get('enable_emotion_detection', True):
                emotion_result = await self.ai_engine.process_text(message, "sentiment")
                context.set_emotion(emotion_result.get('result', {}))
            
            # 意图识别
            intent_result = await self.ai_engine.process_text(message, "intent", 
                                                             {"context": context.to_dict()})
            intent_data = intent_result.get('result', {})
            
            # 实体抽取
            entity_result = await self.ai_engine.process_text(message, "ner")
            entities = entity_result.get('result', {}).get('entities', {})
            
            # 知识库检索
            kb_results = []
            if self.config.get('enable_knowledge_retrieval', True):
                kb_results = await self.kb_manager.search(message, 
                                                         self.config['knowledge_base']['search']['max_results'])
            
            # 状态转移
            new_state = await self._determine_next_state(state_machine.current_state, intent_data, context)
            state_machine.transition_to(new_state)
            
            # 生成回复
            response = await self._generate_response(
                session, context, intent_data, entities, kb_results, state_machine.current_state
            )
            
            # 记录对话历史
            session.add_message("user", message, intent_data, entities)
            session.add_message("assistant", response['reply'], 
                              {"confidence": response.get('confidence', 0.8)}, {})
            
            # 更新统计
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time)
            
            result = {
                "reply": response['reply'],
                "intent": intent_data.get('intent', 'unknown'),
                "confidence": intent_data.get('confidence', 0.0),
                "entities": entities,
                "session_id": session.id,
                "emotion": context.emotion_state,
                "state": state_machine.current_state,
                "suggested_actions": response.get('suggested_actions', []),
                "knowledge_results": kb_results[:2],  # 返回前2个知识结果
                "processing_time_ms": int(processing_time * 1000)
            }
            
            logger.info(f"Processed message for session {session.id}: intent={intent_data.get('intent')}")
            return result
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            return {
                "reply": "抱歉，我现在无法处理您的请求，请稍后再试。",
                "intent": "error",
                "confidence": 0.0,
                "entities": {},
                "session_id": session_id,
                "processing_time_ms": 0
            }
    
    async def _get_or_create_session(self, session_id: str, user_id: str) -> DialogSession:
        """获取或创建会话"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        else:
            # 创建新会话
            session = await self.create_dialog(user_id)
            return session
    
    async def _determine_next_state(self, current_state: str, intent_data: Dict[str, Any], 
                                  context: DialogContext) -> str:
        """确定下一个对话状态"""
        intent = intent_data.get('intent', 'unknown')
        confidence = intent_data.get('confidence', 0.0)
        
        # 状态转移逻辑
        state_transitions = {
            "greeting": {
                "order_query": "problem_solving",
                "product_inquiry": "information_provision",
                "complaint": "empathy_handling",
                "refund": "problem_solving",
                "default": "conversation"
            },
            "conversation": {
                "goodbye": "farewell",
                "order_query": "problem_solving",
                "complaint": "empathy_handling",
                "default": "conversation"
            },
            "problem_solving": {
                "resolved": "confirmation",
                "escalate": "human_handoff",
                "default": "problem_solving"
            },
            "empathy_handling": {
                "calmed": "problem_solving",
                "escalate": "human_handoff",
                "default": "empathy_handling"
            }
        }
        
        next_state_map = state_transitions.get(current_state, {})
        next_state = next_state_map.get(intent, next_state_map.get("default", current_state))
        
        return next_state
    
    async def _generate_response(self, session: DialogSession, context: DialogContext,
                               intent_data: Dict[str, Any], entities: Dict[str, Any],
                               kb_results: List[Dict[str, Any]], current_state: str) -> Dict[str, Any]:
        """生成回复"""
        try:
            # 根据状态和意图选择回复策略
            if current_state == "greeting":
                return await self._generate_greeting_response(context)
            elif current_state == "problem_solving":
                return await self._generate_problem_solving_response(intent_data, entities, kb_results)
            elif current_state == "empathy_handling":
                return await self._generate_empathy_response(context.emotion_state)
            elif current_state == "information_provision":
                return await self._generate_information_response(kb_results)
            else:
                return await self._generate_conversation_response(intent_data, context)
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "reply": "我理解您的问题，让我为您转接人工客服以获得更好的帮助。",
                "confidence": 0.8,
                "suggested_actions": ["联系人工客服", "查看常见问题"]
            }
    
    async def _generate_greeting_response(self, context: DialogContext) -> Dict[str, Any]:
        """生成问候回复"""
        greetings = [
            "您好！欢迎使用AI智能客服系统，我是您的专属助手。请问有什么可以帮助您的吗？",
            "您好！很高兴为您服务。我可以帮您查询订单、解答产品问题、处理投诉等。",
            "您好！我是AI客服助手，随时为您提供专业的服务支持。"
        ]
        
        import random
        return {
            "reply": random.choice(greetings),
            "confidence": 0.95,
            "suggested_actions": ["查询订单", "产品咨询", "投诉建议", "退款申请"]
        }
    
    async def _generate_problem_solving_response(self, intent_data: Dict[str, Any], 
                                               entities: Dict[str, Any], 
                                               kb_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成问题解决回复"""
        intent = intent_data.get('intent', 'unknown')
        
        if kb_results:
            # 使用知识库结果
            best_result = kb_results[0]
            reply = f"关于您的问题，我为您找到了以下解决方案：{best_result.get('content', '')[:200]}..."
        else:
            # 使用模板回复
            templates = {
                "order_query": "我来帮您查询订单状态。请您提供订单号，或者告诉我您想查询哪个订单。",
                "refund": "我理解您需要退款服务。请提供订单号和退款原因，我会立即为您处理。",
                "technical_support": "我来帮您解决技术问题。请详细描述您遇到的具体情况。"
            }
            reply = templates.get(intent, "我来帮您解决这个问题。请提供更多详细信息。")
            
        return {
            "reply": reply,
            "confidence": 0.85,
            "suggested_actions": ["查看详细步骤", "联系技术支持", "转人工客服"]
        }
    
    async def _generate_empathy_response(self, emotion_state: Dict[str, Any]) -> Dict[str, Any]:
        """生成共情回复"""
        sentiment = emotion_state.get('sentiment', 'neutral')
        
        empathy_responses = {
            "negative": [
                "我非常理解您的不满，这确实令人沮丧。让我立即为您解决这个问题。",
                "很抱歉给您带来了不好的体验，我会尽全力帮您处理好这个问题。"
            ],
            "neutral": ["我明白您的意思，让我来帮您解决这个问题。"],
            "positive": ["很高兴为您提供帮助！我会尽快为您解决这个问题。"]
        }
        
        responses = empathy_responses.get(sentiment, empathy_responses["neutral"])
        import random
        return {
            "reply": random.choice(responses),
            "confidence": 0.9,
            "suggested_actions": ["查看解决方案", "转人工客服"]
        }
    
    async def _generate_information_response(self, kb_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成信息提供回复"""
        if kb_results:
            info = kb_results[0]
            reply = f"关于您询问的信息：{info.get('content', '')[:300]}..."
        else:
            reply = "我来为您提供相关信息。请告诉我您具体想了解什么方面？"
            
        return {
            "reply": reply,
            "confidence": 0.88,
            "suggested_actions": ["查看更多详情", "相关产品推荐"]
        }
    
    async def _generate_conversation_response(self, intent_data: Dict[str, Any], 
                                           context: DialogContext) -> Dict[str, Any]:
        """生成一般对话回复"""
        return {
            "reply": "我明白了。还有什么其他问题我可以帮您解决的吗？",
            "confidence": 0.8,
            "suggested_actions": ["查询订单", "产品咨询", "售后服务"]
        }
    
    def get_dialog(self, session_id: str) -> Optional[DialogSession]:
        """获取对话会话"""
        return self.sessions.get(session_id)
    
    async def close_dialog(self, session_id: str) -> bool:
        """关闭对话"""
        try:
            if session_id not in self.sessions:
                return False
                
            session = self.sessions[session_id]
            user_id = session.user_id
            
            # 更新会话状态
            session.close()
            
            # 清理资源
            if session_id in self.contexts:
                del self.contexts[session_id]
            if session_id in self.state_machines:
                del self.state_machines[session_id]
                
            # 从活跃会话中移除
            if user_id in self.active_sessions:
                self.active_sessions[user_id].discard(session_id)
                if not self.active_sessions[user_id]:
                    del self.active_sessions[user_id]
                    
            # 从会话存储中移除
            del self.sessions[session_id]
            
            # 更新统计
            self.stats['active_sessions'] = len(self.sessions)
            
            logger.info(f"Closed dialog session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close dialog {session_id}: {e}")
            return False
    
    async def _cleanup_expired_sessions(self) -> None:
        """清理过期会话"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时检查一次
                
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > self.session_timeout:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self.close_dialog(session_id)
                    
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
            except Exception as e:
                logger.error(f"Session cleanup failed: {e}")
    
    def _update_stats(self, processing_time: float) -> None:
        """更新统计信息"""
        self.stats['total_messages'] += 1
        total_messages = self.stats['total_messages']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (
            (current_avg * (total_messages - 1) + processing_time) / total_messages
        )
        self.stats['last_updated'] = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取对话管理器统计"""
        return {
            **self.stats,
            'active_users': len(self.active_sessions),
            'sessions_per_user': {uid: len(sids) for uid, sids in self.active_sessions.items()},
            'session_states': {
                sid: sm.current_state for sid, sm in self.state_machines.items()
            }
        }