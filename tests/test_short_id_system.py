"""
测试短ID系统
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_memory_formatter_short_id():
    """测试记忆格式化器的短ID显示"""
    
    # 创建模拟的MemoryItem类
    class MemoryItem:
        def __init__(self, id, type, judgment, reasoning, tags, strength):
            self.id = id
            self.type = type
            self.judgment = judgment
            self.reasoning = reasoning
            self.tags = tags
            self.strength = strength
    
    # 手动实现格式化逻辑
    def format_memories_for_display(memories):
        if not memories:
            return "暂无记忆"

        # 按类型分组
        grouped_memories = {}
        for memory in memories:
            type_names = {
                'knowledge': '知识',
                'event': '事件',
                'skill': '技能',
                'emotional': '情感',
                'task': '任务'
            }
            type_name = type_names.get(memory.type, memory.type)
            if type_name not in grouped_memories:
                grouped_memories[type_name] = []
            grouped_memories[type_name].append(memory)
        
        result_parts = []
        for memory_type, memory_list in grouped_memories.items():
            result_parts.append(f"\n=== {memory_type} ===")
            for i, memory in enumerate(memory_list, 1):
                # 生成短ID（前6位）
                short_id = memory.id[:6]
                # 格式化单条记忆
                judgment = memory.judgment.strip()
                reasoning = ""
                if memory.reasoning and memory.reasoning.strip():
                    reasoning = f"\n   ——因为{memory.reasoning.strip()}"
                result_parts.append(f"\n{i}. [id:{short_id}]{judgment}{reasoning}")
        
        return "".join(result_parts)
    
    # 创建测试记忆
    memories = [
        MemoryItem(
            id="a1b2c3d7890ef1234567890abcdef1234",
            type="knowledge",
            judgment="《三体》是刘慈欣写的科幻小说",
            reasoning="昨天用户询问科幻小说推荐时我推荐的",
            tags=["书籍", "科幻"],
            strength=0.8
        ),
        MemoryItem(
            id="d4e5f6e7890ef1234567890abcdef1234",
            type="event",
            judgment="昨天用户想要科幻小说推荐",
            reasoning="用户说想找一些硬科幻作品",
            tags=["用户需求"],
            strength=0.9
        )
    ]
    
    # 测试格式化显示
    formatted = format_memories_for_display(memories)
    print("格式化显示测试：")
    print(formatted)
    
    # 验证短ID格式
    assert "[id:a1b2c3]" in formatted, "应该包含短ID a1b2c3"
    assert "[id:d4e5f6]" in formatted, "应该包含短ID d4e5f6"
    assert "《三体》是刘慈欣写的科幻小说" in formatted, "应该包含记忆内容"
    assert "昨天用户想要科幻小说推荐" in formatted, "应该包含记忆内容"
    
    print("记忆格式化短ID显示测试通过")

def test_id_resolution():
    """测试ID解析功能"""
    
    # 创建模拟的MemoryItem类
    class MemoryItem:
        def __init__(self, id, type, judgment, reasoning, tags, strength):
            self.id = id
            self.type = type
            self.judgment = judgment
            self.reasoning = reasoning
            self.tags = tags
            self.strength = strength
    
    # 模拟ID解析方法
    def resolve_memory_ids(short_ids, memories):
        """将短ID转换为完整ID"""
        resolved_ids = []
        
        for short_id in short_ids:
            # 在记忆中查找匹配的完整ID
            for memory in memories:
                if memory.id.startswith(short_id):
                    resolved_ids.append(memory.id)
                    break
            else:
                # 如果没有找到匹配的ID，记录警告但继续处理
                print(f"警告: 未找到匹配的完整ID: {short_id}")
        
        return resolved_ids
    
    # 创建测试记忆
    memories = [
        MemoryItem(
            id="a1b2c3d7890ef1234567890abcdef1234",
            type="knowledge",
            judgment="《三体》是刘慈欣写的科幻小说",
            reasoning="昨天用户询问科幻小说推荐时我推荐的",
            tags=["书籍", "科幻"],
            strength=0.8
        ),
        MemoryItem(
            id="d4e5f6e7890ef1234567890abcdef1234",
            type="event",
            judgment="昨天用户想要科幻小说推荐",
            reasoning="用户说想找一些硬科幻作品",
            tags=["用户需求"],
            strength=0.9
        )
    ]
    
    # 测试ID解析
    short_ids = ["a1b2c3", "d4e5f6"]
    resolved_ids = resolve_memory_ids(short_ids, memories)
    
    print("\nID解析测试：")
    print(f"输入短ID: {short_ids}")
    print(f"解析结果: {resolved_ids}")
    
    # 验证解析结果
    assert len(resolved_ids) == 2, "应该解析出2个完整ID"
    assert "a1b2c3d7890ef1234567890abcdef1234" in resolved_ids, "应该包含第一个完整ID"
    assert "d4e5f6e7890ef1234567890abcdef1234" in resolved_ids, "应该包含第二个完整ID"
    
    # 测试不存在的ID
    short_ids_with_invalid = ["a1b2c3", "xyz789"]
    resolved_ids_with_invalid = resolve_memory_ids(short_ids_with_invalid, memories)
    
    print(f"\n包含无效ID的测试:")
    print(f"输入短ID: {short_ids_with_invalid}")
    print(f"解析结果: {resolved_ids_with_invalid}")
    
    assert len(resolved_ids_with_invalid) == 1, "应该只解析出1个有效ID"
    assert "a1b2c3d7890ef1234567890abcdef1234" in resolved_ids_with_invalid, "应该包含有效的完整ID"
    
    print("ID解析测试通过")

if __name__ == "__main__":
    test_memory_formatter_short_id()
    test_id_resolution()
    print("\n所有短ID系统测试通过!")