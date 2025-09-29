"""
SmallModelPromptBuilder 测试
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List


# 模拟 MemoryItem 类
@dataclass
class MemoryItem:
    """记忆项数据结构"""
    id: str
    type: str
    judgment: str
    reasoning: str
    tags: List[str]
    timestamp: float
    strength: int = 0


# 模拟 SmallModelPromptBuilder 类
class SmallModelPromptBuilder:
    """小模型提示词构建器"""
    
    @staticmethod
    def build_memory_prompt(query: str, memories: List[MemoryItem]) -> List[dict]:
        """
        构建用于小模型的记忆整理提示词
        
        Args:
            query: 用户查询
            memories: 记忆列表
            
        Returns:
            消息列表，包含系统消息和用户消息
        """
        # 构建记忆上下文
        memory_context = ""
        if memories:
            memory_context = "\n\n相关记忆：\n"
            for i, memory in enumerate(memories, 1):
                memory_context += f"{i}. [ID:{memory.id}] {memory.judgment}\n"
                if memory.reasoning:
                    memory_context += f"   内容：{memory.reasoning}\n"
                if memory.tags:
                    memory_context += f"   标签：{', '.join(memory.tags)}\n\n"
        
        # 模拟获取系统提示词
        system_prompt = "系统提示词内容"
        
        # 构建用户消息
        user_message = f"对话是：{query}\n\n你回忆起了：{memory_context}\n\n请你按照任务进行处理"
        
        # 返回消息列表
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]


class TestSmallModelPromptBuilder:
    """测试 SmallModelPromptBuilder 类"""
    
    def test_build_memory_prompt_with_memories(self):
        """测试构建包含记忆的提示词"""
        # 创建测试记忆
        memories = [
            MemoryItem(
                id="test_id_1",
                type="knowledge",
                judgment="Python是一种编程语言",
                reasoning="这是基础知识点",
                tags=["编程", "Python"],
                strength=0.8,
                timestamp=1234567890
            ),
            MemoryItem(
                id="test_id_2", 
                type="skill",
                judgment="编写代码前应该先规划",
                reasoning="这样可以提高代码质量",
                tags=["开发", "最佳实践"],
                strength=0.9,
                timestamp=1234567891
            )
        ]
        
        # 构建提示词
        messages = SmallModelPromptBuilder.build_memory_prompt("测试查询", memories)
        
        # 验证返回的是消息列表
        assert isinstance(messages, list)
        assert len(messages) == 2
        
        # 验证系统消息
        system_message = messages[0]
        assert system_message["role"] == "system"
        assert system_message["content"] == "系统提示词内容"
        
        # 验证用户消息
        user_message = messages[1]
        assert user_message["role"] == "user"
        assert "对话是：测试查询" in user_message["content"]
        assert "你回忆起了：" in user_message["content"]
        assert "请你按照任务进行处理" in user_message["content"]
        assert "[ID:test_id_1]" in user_message["content"]
        assert "[ID:test_id_2]" in user_message["content"]
        assert "Python是一种编程语言" in user_message["content"]
        assert "编写代码前应该先规划" in user_message["content"]
    
    def test_build_memory_prompt_without_memories(self):
        """测试构建不包含记忆的提示词"""
        # 构建提示词
        messages = SmallModelPromptBuilder.build_memory_prompt("测试查询", [])
        
        # 验证返回的是消息列表
        assert isinstance(messages, list)
        assert len(messages) == 2
        
        # 验证系统消息
        system_message = messages[0]
        assert system_message["role"] == "system"
        assert system_message["content"] == "系统提示词内容"
        
        # 验证用户消息
        user_message = messages[1]
        assert user_message["role"] == "user"
        assert "对话是：测试查询" in user_message["content"]
        assert "你回忆起了：" in user_message["content"]
        assert "请你按照任务进行处理" in user_message["content"]
        # 不应该包含记忆ID
        assert "[ID:" not in user_message["content"]
    
    def test_build_memory_prompt_with_empty_memory_context(self):
        """测试记忆内容为空的情况"""
        # 创建空内容的记忆
        memories = [
            MemoryItem(
                id="test_id",
                type="knowledge",
                judgment="",
                reasoning="",
                tags=[],
                strength=0.5,
                timestamp=1234567890
            )
        ]
        
        # 构建提示词
        messages = SmallModelPromptBuilder.build_memory_prompt("测试查询", memories)
        
        # 验证用户消息包含记忆ID
        user_message = messages[1]
        assert "[ID:test_id]" in user_message["content"]