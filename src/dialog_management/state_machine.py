"""对话状态机 - Claude Code风格"""
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class DialogState(Enum):
    """对话状态枚举"""
    GREETING = "greeting"
    CONVERSATION = "conversation"
    PROBLEM_SOLVING = "problem_solving"
    EMPATHY_HANDLING = "empathy_handling"
    INFORMATION_PROVISION = "information_provision"
    CONFIRMATION = "confirmation"
    HUMAN_HANDOFF = "human_handoff"
    FAREWELL = "farewell"

class StateTransition:
    """状态转移模型"""
    def __init__(self, from_state: DialogState, to_state: DialogState, 
                 condition: Callable[[Dict[str, Any]], bool], 
                 action: Callable[[], None] = None):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition
        self.action = action

class DialogStateMachine:
    """对话状态机"""
    
    def __init__(self, initial_state: str = "greeting"):
        self.states = {state: {} for state in DialogState}
        self.current_state = DialogState(initial_state)
        self.previous_state = None
        self.state_history = []
        self.transition_rules = []
        
        # 注册状态转移规则
        self._register_transition_rules()
        
        logger.info(f"Dialog state machine initialized with state: {initial_state}")
        
    def _register_transition_rules(self) -> None:
        """注册状态转移规则"""
        # 从问候状态转移
        self.add_transition_rule(
            DialogState.GREETING, DialogState.CONVERSATION,
            lambda ctx: ctx.get('intent') in ['smalltalk', 'unknown'],
            lambda: logger.info("Transition: greeting -> conversation")
        )
        
        self.add_transition_rule(
            DialogState.GREETING, DialogState.PROBLEM_SOLVING,
            lambda ctx: ctx.get('intent') in ['order_query', 'refund', 'technical_support'],
            lambda: logger.info("Transition: greeting -> problem_solving")
        )
        
        self.add_transition_rule(
            DialogState.GREETING, DialogState.EMPATHY_HANDLING,
            lambda ctx: ctx.get('intent') == 'complaint' or ctx.get('emotion') == 'negative',
            lambda: logger.info("Transition: greeting -> empathy_handling")
        )
        
        self.add_transition_rule(
            DialogState.GREETING, DialogState.INFORMATION_PROVISION,
            lambda ctx: ctx.get('intent') == 'product_inquiry',
            lambda: logger.info("Transition: greeting -> information_provision")
        )
        
        # 从对话状态转移
        self.add_transition_rule(
            DialogState.CONVERSATION, DialogState.PROBLEM_SOLVING,
            lambda ctx: ctx.get('intent') in ['order_query', 'refund', 'technical_support'],
            lambda: logger.info("Transition: conversation -> problem_solving")
        )
        
        self.add_transition_rule(
            DialogState.CONVERSATION, DialogState.FAREWELL,
            lambda ctx: ctx.get('intent') == 'goodbye',
            lambda: logger.info("Transition: conversation -> farewell")
        )
        
        # 从问题解决状态转移
        self.add_transition_rule(
            DialogState.PROBLEM_SOLVING, DialogState.CONFIRMATION,
            lambda ctx: ctx.get('problem_resolved', False),
            lambda: logger.info("Transition: problem_solving -> confirmation")
        )
        
        self.add_transition_rule(
            DialogState.PROBLEM_SOLVING, DialogState.HUMAN_HANDOFF,
            lambda ctx: ctx.get('escalate', False) or ctx.get('intent') == 'human_transfer',
            lambda: logger.info("Transition: problem_solving -> human_handoff")
        )
        
        self.add_transition_rule(
            DialogState.PROBLEM_SOLVING, DialogState.CONVERSATION,
            lambda ctx: ctx.get('awaiting_user_input', True),
            lambda: logger.info("Transition: problem_solving -> conversation")
        )
        
        # 从共情处理状态转移
        self.add_transition_rule(
            DialogState.EMPATHY_HANDLING, DialogState.PROBLEM_SOLVING,
            lambda ctx: ctx.get('user_calmed', False) and ctx.get('has_problem', True),
            lambda: logger.info("Transition: empathy_handling -> problem_solving")
        )
        
        self.add_transition_rule(
            DialogState.EMPATHY_HANDLING, DialogState.HUMAN_HANDOFF,
            lambda ctx: ctx.get('escalate', False),
            lambda: logger.info("Transition: empathy_handling -> human_handoff")
        )
        
        self.add_transition_rule(
            DialogState.EMPATHY_HANDLING, DialogState.CONVERSATION,
            lambda ctx: ctx.get('user_calmed', False) and not ctx.get('has_problem', True),
            lambda: logger.info("Transition: empathy_handling -> conversation")
        )
        
        # 从信息提供状态转移
        self.add_transition_rule(
            DialogState.INFORMATION_PROVISION, DialogState.CONVERSATION,
            lambda ctx: True,  # 总是可以回到对话状态
            lambda: logger.info("Transition: information_provision -> conversation")
        )
        
        # 从确认状态转移
        self.add_transition_rule(
            DialogState.CONFIRMATION, DialogState.FAREWELL,
            lambda ctx: ctx.get('satisfied', True),
            lambda: logger.info("Transition: confirmation -> farewell")
        )
        
        self.add_transition_rule(
            DialogState.CONFIRMATION, DialogState.PROBLEM_SOLVING,
            lambda ctx: not ctx.get('satisfied', True),
            lambda: logger.info("Transition: confirmation -> problem_solving")
        )
        
        # 从人工转接状态转移
        self.add_transition_rule(
            DialogState.HUMAN_HANDOFF, DialogState.FAREWELL,
            lambda ctx: ctx.get('transfer_completed', True),
            lambda: logger.info("Transition: human_handoff -> farewell")
        )
        
        # 从告别状态转移
        self.add_transition_rule(
            DialogState.FAREWELL, DialogState.GREETING,
            lambda ctx: ctx.get('new_conversation', False),
            lambda: logger.info("Transition: farewell -> greeting")
        )
    
    def add_transition_rule(self, from_state: DialogState, to_state: DialogState,
                          condition: Callable[[Dict[str, Any]], bool],
                          action: Callable[[], None] = None) -> None:
        """添加状态转移规则"""
        transition = StateTransition(from_state, to_state, condition, action)
        self.transition_rules.append(transition)
        
        # 注册到源状态
        if to_state not in self.states[from_state]:
            self.states[from_state][to_state] = []
        self.states[from_state][to_state].append(transition)
    
    def transition_to(self, new_state: str) -> bool:
        """转移到新状态"""
        try:
            new_state_enum = DialogState(new_state)
            current_context = {
                'current_state': self.current_state.value,
                'previous_state': self.previous_state.value if self.previous_state else None,
                'state_history': [s.value for s in self.state_history]
            }
            
            # 检查是否有允许的规则
            allowed_transitions = self.states[self.current_state].get(new_state_enum, [])
            valid_transition = None
            
            for transition in allowed_transitions:
                if transition.condition(current_context):
                    valid_transition = transition
                    break
            
            if not valid_transition and new_state_enum != self.current_state:
                logger.warning(f"Invalid transition attempted: {self.current_state.value} -> {new_state}")
                return False
            
            # 执行转移动作
            if valid_transition and valid_transition.action:
                valid_transition.action()
            
            # 更新状态
            self.previous_state = self.current_state
            self.current_state = new_state_enum
            self.state_history.append(self.previous_state)
            
            # 限制历史长度
            if len(self.state_history) > 10:
                self.state_history = self.state_history[-10:]
                
            logger.info(f"State transitioned: {self.previous_state.value} -> {self.current_state.value}")
            return True
            
        except ValueError as e:
            logger.error(f"Invalid state name: {new_state}")
            return False
        except Exception as e:
            logger.error(f"State transition failed: {e}")
            return False
    
    def can_transition_to(self, state: str) -> bool:
        """检查是否可以转移到指定状态"""
        try:
            target_state = DialogState(state)
            transitions = self.states[self.current_state].get(target_state, [])
            
            for transition in transitions:
                # 简化的上下文检查
                dummy_context = {'intent': 'unknown'}
                if transition.condition(dummy_context):
                    return True
                    
            return target_state == self.current_state
            
        except ValueError:
            return False
    
    def get_available_transitions(self) -> List[str]:
        """获取当前可用的状态转移"""
        available = []
        
        for target_state in self.states[self.current_state]:
            transitions = self.states[self.current_state][target_state]
            for transition in transitions:
                dummy_context = {'intent': 'unknown'}
                if transition.condition(dummy_context):
                    available.append(target_state.value)
                    break
                    
        return available
    
    def get_state_info(self) -> Dict[str, Any]:
        """获取状态机信息"""
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'available_transitions': self.get_available_transitions(),
            'state_history': [s.value for s in self.state_history],
            'total_transitions': len(self.state_history)
        }
    
    def reset(self, initial_state: str = "greeting") -> None:
        """重置状态机"""
        self.current_state = DialogState(initial_state)
        self.previous_state = None
        self.state_history.clear()
        
        logger.info(f"State machine reset to: {initial_state}")