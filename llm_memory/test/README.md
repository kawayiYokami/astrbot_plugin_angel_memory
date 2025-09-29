# 测试套件

## 测试结构

```
test/
├── conftest.py              # pytest配置和fixtures
├── test_data_models.py      # 数据模型测试
├── test_memory_handlers.py  # 记忆处理器测试
├── test_cognitive_service.py # 认知服务测试
├── test_memory_manager.py   # 记忆管理器测试
└── test_integration.py      # 集成测试
```

## 运行测试

### 运行所有测试

```bash
pytest llm_memory/test/
```

### 运行特定测试文件

```bash
pytest llm_memory/test/test_data_models.py
```

### 运行特定测试类

```bash
pytest llm_memory/test/test_data_models.py::TestBaseMemory
```

### 运行特定测试方法

```bash
pytest llm_memory/test/test_data_models.py::TestBaseMemory::test_create_knowledge_memory
```

### 查看详细输出

```bash
pytest llm_memory/test/ -v
```

### 查看测试覆盖率

```bash
pytest llm_memory/test/ --cov=llm_memory --cov-report=html
```

## 测试覆盖范围

### test_data_models.py
- BaseMemory类的创建和初始化
- 记忆类型枚举
- 字典序列化和反序列化
- 语义核心生成
- 标签解析
- 关联字段处理

### test_memory_handlers.py
- MemoryHandler类的基本功能
- 记忆的存储和检索
- 记忆类型过滤
- MemoryHandlerFactory工厂模式

### test_cognitive_service.py
- CognitiveService初始化
- 通用记忆接口
- 各类型记忆的专用接口
- 综合回忆功能
- 记忆巩固
- 反馈处理
- 存储路径管理

### test_memory_manager.py
- 记忆强化
- 记忆巩固
- 综合回忆和去重
- 记忆合并
- 反馈处理
- 从元数据构建记忆

### test_integration.py
- 完整工作流测试
- 记忆巩固流程
- 记忆合并流程
- 多类型记忆检索
- 关联建立
- 强度累积
- 批量操作
- 边界情况

## 注意事项

1. 所有测试使用临时目录，不会影响实际数据
2. 每个测试独立运行，互不干扰
3. 测试会自动清理临时数据
4. 如果测试失败，检查是否缺少依赖包

## 依赖

确保已安装以下包：

```bash
pip install pytest pytest-cov
```
