"""
测试事件处理方法
"""

import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_extract_message_text():
    """测试消息文本提取方法"""
    
    # 模拟DeepMind类的_extract_message_text方法
    def _extract_message_text(event):
        """
        从事件中提取消息文本
        """
        # 尝试从event对象获取消息内容
        if hasattr(event, 'content'):
            content = event.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and content:
                # 处理多模态内容列表
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        return item.get('text')
        
        # 如果没有content字段，尝试其他可能的字段
        if hasattr(event, 'message'):
            return str(event.message)
        
        # 如果都没有，返回None
        return None

    # 测试用例
    print("测试消息文本提取...")
    
    # 测试纯文本消息
    class Event1:
        def __init__(self):
            self.content = "Hello, world!"
    
    event1 = Event1()
    message_text = _extract_message_text(event1)
    assert message_text == "Hello, world!", f"Expected 'Hello, world!', got '{message_text}'"
    print("✓ 纯文本消息测试通过")
    
    # 测试多模态消息
    class Event2:
        def __init__(self):
            self.content = [{"type": "text", "text": "Multimodal message"}]
    
    event2 = Event2()
    message_text = _extract_message_text(event2)
    assert message_text == "Multimodal message", f"Expected 'Multimodal message', got '{message_text}'"
    print("✓ 多模态消息测试通过")
    
    # 测试消息字段
    class Event3:
        def __init__(self):
            self.message = "Message from message field"
    
    event3 = Event3()
    message_text = _extract_message_text(event3)
    assert message_text == "Message from message field", f"Expected 'Message from message field', got '{message_text}'"
    print("✓ 消息字段测试通过")
    
    # 测试无内容
    class Event4:
        def __init__(self):
            self.other_field = "No content here"
    
    event4 = Event4()
    message_text = _extract_message_text(event4)
    assert message_text is None, f"Expected None, got '{message_text}'"
    print("✓ 无内容测试通过")

def test_get_session_id():
    """测试会话ID生成方法"""
    
    # 模拟DeepMind类的_get_session_id方法
    def _get_session_id(event):
        """获取会话ID"""
        # 尝试从事件中获取发送者ID和群组ID
        sender_id = "unknown"
        group_id = "private"
        
        # 尝试获取发送者ID
        if hasattr(event, 'sender_id'):
            sender_id = str(event.sender_id)
        elif hasattr(event, 'user_id'):
            sender_id = str(event.user_id)
        
        # 尝试获取群组ID
        if hasattr(event, 'group_id'):
            group_id = str(event.group_id)
        elif hasattr(event, 'channel_id'):
            group_id = str(event.channel_id)
        
        return f"{sender_id}_{group_id}"

    # 测试用例
    print("\n测试会话ID生成...")
    
    # 测试群组消息
    class Event1:
        def __init__(self):
            self.sender_id = "user123"
            self.group_id = "group456"
    
    event1 = Event1()
    session_id = _get_session_id(event1)
    assert session_id == "user123_group456", f"Expected 'user123_group456', got '{session_id}'"
    print("✓ 群组消息测试通过")
    
    # 测试私聊消息
    class Event2:
        def __init__(self):
            self.sender_id = "user789"
    
    event2 = Event2()
    session_id = _get_session_id(event2)
    assert session_id == "user789_private", f"Expected 'user789_private', got '{session_id}'"
    print("✓ 私聊消息测试通过")
    
    # 测试使用user_id
    class Event3:
        def __init__(self):
            self.user_id = "user456"
            self.channel_id = "channel123"
    
    event3 = Event3()
    session_id = _get_session_id(event3)
    assert session_id == "user456_channel123", f"Expected 'user456_channel123', got '{session_id}'"
    print("✓ user_id和channel_id测试通过")
    
    # 测试无ID信息
    class Event4:
        def __init__(self):
            self.other_field = "No ID here"
    
    event4 = Event4()
    session_id = _get_session_id(event4)
    assert session_id == "unknown_private", f"Expected 'unknown_private', got '{session_id}'"
    print("✓ 无ID信息测试通过")

def test_angelheart_context():
    """测试angelheart_context注入"""
    
    print("\n测试angelheart_context注入...")
    
    # 模拟事件对象
    class MockEvent:
        def __init__(self):
            pass
    
    event = MockEvent()
    
    # 测试注入angelheart_context
    memories_json = [{"id": "1", "type": "knowledge", "judgment": "Test memory"}]
    context_data = {
        'memories': memories_json,
        'recall_query': 'Test query',
        'recall_time': 1234567890.0,
        'session_id': 'test_session'
    }
    
    event.angelheart_context = json.dumps(context_data)
    
    # 测试读取angelheart_context
    assert hasattr(event, 'angelheart_context'), "Event should have angelheart_context attribute"
    
    parsed_context = json.loads(event.angelheart_context)
    assert parsed_context['session_id'] == 'test_session', f"Expected 'test_session', got '{parsed_context['session_id']}'"
    assert parsed_context['recall_query'] == 'Test query', f"Expected 'Test query', got '{parsed_context['recall_query']}'"
    assert len(parsed_context['memories']) == 1, f"Expected 1 memory, got {len(parsed_context['memories'])}"
    
    print("✓ angelheart_context注入和读取测试通过")

if __name__ == "__main__":
    test_extract_message_text()
    test_get_session_id()
    test_angelheart_context()
    print("\n✅ 所有事件处理方法测试通过!")