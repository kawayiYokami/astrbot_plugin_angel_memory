"""
测试事件处理逻辑
"""

import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模拟AstrBot事件类
class MockAstrMessageEvent:
    def __init__(self, content, sender_id="test_user", group_id=None):
        self.content = content
        self.sender_id = sender_id
        self.group_id = group_id

# 模拟ProviderRequest类
class MockProviderRequest:
    def __init__(self):
        pass

# 模拟LLMResponse类
class MockLLMResponse:
    def __init__(self):
        pass

# 模拟AstrBotConfig类
class MockAstrBotConfig:
    def __init__(self):
        pass

# 模拟Context类
class MockContext:
    def __init__(self):
        pass

def test_event_processing():
    """测试事件处理流程"""
    
    # 动态导入模块以避免依赖问题
    import importlib.util
    
    # 加载配置模块
    config_spec = importlib.util.spec_from_file_location(
        "config", "core/config.py"
    )
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    
    # 加载日志模块
    logger_spec = importlib.util.spec_from_file_location(
        "logger", "core/logger.py"
    )
    logger_module = importlib.util.module_from_spec(logger_spec)
    logger_spec.loader.exec_module(logger_module)
    
    # 加载DeepMind模块
    deepmind_spec = importlib.util.spec_from_file_location(
        "deepmind", "core/deepmind.py"
    )
    deepmind_module = importlib.util.module_from_spec(deepmind_spec)
    
    # 模拟依赖
    deepmind_module.logger = logger_module.logger
    deepmind_module.CognitiveService = type('CognitiveService', (), {
        '__init__': lambda self, logger=None: None,
        'chained_recall': lambda self, query: [],
        'feedback': lambda self, **kwargs: None
    })
    
    deepmind_spec.loader.exec_module(deepmind_module)
    
    # 创建配置和日志实例
    config = config_module.MemoryConfig(MockAstrBotConfig(), "test_plugin")
    logger = logger_module.MemoryLogger()
    
    # 创建DeepMind实例
    deepmind = deepmind_module.DeepMind(config, logger)
    
    # 测试消息文本提取
    print("测试消息文本提取...")
    
    # 测试纯文本消息
    event1 = MockAstrMessageEvent("Hello, world!")
    message_text = deepmind._extract_message_text(event1)
    assert message_text == "Hello, world!", f"Expected 'Hello, world!', got '{message_text}'"
    
    # 测试多模态消息
    event2 = MockAstrMessageEvent([{"type": "text", "text": "Multimodal message"}])
    message_text = deepmind._extract_message_text(event2)
    assert message_text == "Multimodal message", f"Expected 'Multimodal message', got '{message_text}'"
    
    # 测试会话ID生成
    print("测试会话ID生成...")
    
    event3 = MockAstrMessageEvent("Test", sender_id="user123", group_id="group456")
    session_id = deepmind._get_session_id(event3)
    assert session_id == "user123_group456", f"Expected 'user123_group456', got '{session_id}'"
    
    event4 = MockAstrMessageEvent("Test", sender_id="user789")
    session_id = deepmind._get_session_id(event4)
    assert session_id == "user789_private", f"Expected 'user789_private', got '{session_id}'"
    
    print("✓ 所有事件处理测试通过!")

if __name__ == "__main__":
    test_event_processing()