"""NLP处理器 - Claude Code风格"""
import re
import logging
from typing import Dict, List, Any, Optional
import jieba
import jieba.posseg as pseg
from snownlp import SnowNLP
from collections import Counter

logger = logging.getLogger(__name__)

class NLPProcessor:
    """自然语言处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.spell_checker = SpellChecker()
        self.synonym_expander = SynonymExpander()
        self.stop_words = self._load_stop_words()
        
        # 配置jieba
        jieba.initialize()
        if self.config.get('custom_dict_path'):
            jieba.load_userdict(self.config['custom_dict_path'])
            
        logger.info("NLP Processor initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'enable_spell_check': True,
            'synonym_expansion': True,
            'max_text_length': 512,
            'remove_stop_words': True,
            'segmentation_method': 'jieba',  # jieba, pkuseg, thulac
            'pos_tagging': False,
            'custom_dict_path': None
        }
    
    def _load_stop_words(self) -> set:
        """加载停用词表"""
        default_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '我们',
            '你们', '他们', '这', '那', '哪', '怎么', '什么', '为什么', '如何'
        }
        return default_stop_words
    
    async def preprocess(self, text: str, task_type: str = "general") -> Dict[str, Any]:
        """文本预处理"""
        try:
            # 长度检查
            if len(text) > self.config.get('max_text_length', 512):
                text = text[:self.config['max_text_length']]
                logger.warning(f"Text truncated to {self.config['max_text_length']} characters")
            
            # 拼写检查
            if self.config.get('enable_spell_check', True):
                text = await self.spell_checker.correct(text)
            
            # 文本清洗
            text = self._clean_text(text)
            
            # 分词和标注
            segmentation_result = await self._segment_text(text, task_type)
            
            # 同义词扩展
            if self.config.get('synonym_expansion', True):
                segmentation_result = await self.synonym_expander.expand(segmentation_result)
            
            # 去除停用词
            if self.config.get('remove_stop_words', True):
                segmentation_result = self._remove_stop_words(segmentation_result)
            
            return {
                'original_text': text,
                'processed_text': segmentation_result['text'],
                'tokens': segmentation_result['tokens'],
                'keywords': segmentation_result.get('keywords', []),
                'entities': segmentation_result.get('entities', []),
                'task_type': task_type,
                'processing_steps': [
                    'length_check', 'spell_check', 'cleaning', 
                    'segmentation', 'synonym_expansion', 'stop_word_removal'
                ]
            }
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return {
                'original_text': text,
                'processed_text': text,
                'tokens': [],
                'error': str(e)
            }
    
    def _clean_text(self, text: str) -> str:
        """文本清洗"""
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # 标准化标点符号
        text = re.sub(r'[\uff01-\uff5e]', lambda x: chr(ord(x.group(0)) - 0xfee0), text)
        
        return text
    
    async def _segment_text(self, text: str, task_type: str) -> Dict[str, Any]:
        """文本分词"""
        method = self.config.get('segmentation_method', 'jieba')
        
        if method == 'jieba':
            return await self._jieba_segment(text, task_type)
        else:
            # 默认使用jieba
            return await self._jieba_segment(text, task_type)
    
    async def _jieba_segment(self, text: str, task_type: str) -> Dict[str, Any]:
        """使用jieba分词"""
        # 基础分词
        words = jieba.lcut(text)
        
        # 词性标注
        pos_tags = []
        if self.config.get('pos_tagging', False):
            pos_tags = [(word, flag) for word, flag in pseg.cut(text)]
        
        # 提取关键词
        keywords = await self._extract_keywords(text, words)
        
        # 实体识别（简化版）
        entities = await self._extract_simple_entities(text)
        
        return {
            'text': ' '.join(words),
            'tokens': words,
            'pos_tags': pos_tags,
            'keywords': keywords,
            'entities': entities
        }
    
    async def _extract_keywords(self, text: str, tokens: List[str]) -> List[str]:
        """提取关键词"""
        try:
            # 使用SnowNLP提取关键词
            s = SnowNLP(text)
            keywords = s.keywords(10)  # 提取前10个关键词
            
            # 结合TF-IDF思想，统计词频
            word_freq = Counter([token for token in tokens if len(token) > 1])
            high_freq_words = [word for word, freq in word_freq.most_common(5) if freq > 1]
            
            # 合并结果
            all_keywords = list(set(keywords + high_freq_words))
            return all_keywords[:10]  # 返回前10个
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return tokens[:5]  #  fallback: 返回前5个词
    
    async def _extract_simple_entities(self, text: str) -> List[Dict[str, str]]:
        """简单实体识别"""
        entities = []
        
        # 手机号模式
        phone_pattern = r'1[3-9]\d{9}'
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            entities.append({'text': phone, 'type': 'PHONE', 'start': text.find(phone)})
        
        # 邮箱模式
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            entities.append({'text': email, 'type': 'EMAIL', 'start': text.find(email)})
        
        # 数字模式（可能是订单号、金额等）
        number_pattern = r'\b\d{6,}\b'
        numbers = re.findall(number_pattern, text)
        for number in numbers:
            entities.append({'text': number, 'type': 'NUMBER', 'start': text.find(number)})
        
        return entities
    
    def _remove_stop_words(self, segmentation_result: Dict[str, Any]) -> Dict[str, Any]:
        """去除停用词"""
        tokens = segmentation_result['tokens']
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        segmentation_result['tokens'] = filtered_tokens
        segmentation_result['text'] = ' '.join(filtered_tokens)
        return segmentation_result

class SpellChecker:
    """拼写检查器"""
    
    def __init__(self):
        self.common_corrections = {
            '么': '吗', '那': '哪', '坐': '座', '在': '再',
            '以': '已', '做': '作', '象': '像', '记': '纪'
        }
    
    async def correct(self, text: str) -> str:
        """纠正拼写错误"""
        corrected_text = text
        
        for wrong, right in self.common_corrections.items():
            corrected_text = corrected_text.replace(wrong, right)
            
        return corrected_text

class SynonymExpander:
    """同义词扩展器"""
    
    def __init__(self):
        self.synonym_dict = {
            '手机': ['电话', '移动电话', '智能机'],
            '电脑': ['计算机', '笔记本', '台式机'],
            '问题': ['疑问', '难题', '故障'],
            '帮助': ['协助', '支援', '支持'],
            '购买': ['采购', '选购', '下单']
        }
    
    async def expand(self, segmentation_result: Dict[str, Any]) -> Dict[str, Any]:
        """扩展同义词"""
        tokens = segmentation_result['tokens']
        expanded_tokens = tokens.copy()
        
        for token in tokens:
            if token in self.synonym_dict:
                # 添加前3个同义词
                synonyms = self.synonym_dict[token][:3]
                expanded_tokens.extend(synonyms)
                
        segmentation_result['tokens'] = expanded_tokens
        segmentation_result['text'] = ' '.join(expanded_tokens)
        return segmentation_result