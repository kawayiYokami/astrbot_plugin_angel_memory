"""angel_remember 兼容模型摊平的参数（#89）。

长上下文 + 多工具并存的工具循环里，模型偶发把 memory 的子字段提到顶层，
并把 memory 写成字符串。框架原样展开 kwargs 调 run()，多出来的键会在参数
绑定阶段直接抛 TypeError，记忆静默丢失。这里钉住掰回嵌套形状的收口逻辑。
"""

import pytest
from astrbot_plugin_angel_memory.tools.angel_remember import CoreMemoryRememberTool


def test_flattened_params_are_folded_back_into_memory():
    """memory 是字符串、子字段在顶层时，掰回嵌套形状。"""
    memory = CoreMemoryRememberTool._normalize_memory(
        "用户偏好简短回复",
        {"memory_type": "knowledge", "reasoning": "多次要求", "tags": ["偏好", "回复风格"]},
    )

    assert memory == {
        "judgment": "用户偏好简短回复",
        "memory_type": "knowledge",
        "reasoning": "多次要求",
        "tags": ["偏好", "回复风格"],
    }


def test_nested_params_are_left_untouched():
    """本来就嵌套正确时原样返回，不额外加工。"""
    source = {"judgment": "判断句", "memory_type": "skill", "tags": ["锚点"]}

    memory = CoreMemoryRememberTool._normalize_memory(source, {})

    assert memory == source


def test_existing_memory_value_wins_over_flattened_duplicate():
    """memory 里已有非空值时，不被顶层同名字段覆盖。"""
    memory = CoreMemoryRememberTool._normalize_memory(
        {"judgment": "保留这句", "tags": ["锚点"]},
        {"judgment": "顶层那句", "memory_type": "event"},
    )

    assert memory["judgment"] == "保留这句"
    assert memory["memory_type"] == "event"


def test_empty_memory_value_is_filled_from_flattened_field():
    """memory 里字段是空值时，用顶层的补上。"""
    memory = CoreMemoryRememberTool._normalize_memory(
        {"judgment": "判断句", "tags": []},
        {"tags": ["补上的"]},
    )

    assert memory["tags"] == ["补上的"]


def test_memory_none_with_all_fields_on_top_level():
    """memory 整个缺失、子字段全在顶层时也能拼成完整对象。"""
    memory = CoreMemoryRememberTool._normalize_memory(
        None,
        {"judgment": "判断句", "memory_type": "event", "tags": ["锚点"]},
    )

    assert memory == {"judgment": "判断句", "memory_type": "event", "tags": ["锚点"]}


def test_non_dict_non_str_memory_becomes_empty_dict():
    """memory 传了既不是对象也不是字符串的值时，收成空对象交给后续校验拒绝。"""
    memory = CoreMemoryRememberTool._normalize_memory(12345, {})

    assert memory == {}


@pytest.mark.asyncio
async def test_run_accepts_flattened_params_without_type_error():
    """模型摊平参数时 run() 不再因多余关键字抛 TypeError。"""
    tool = CoreMemoryRememberTool()

    # 不带 plugin_context 的事件：参数校验能过，停在校验之后取上下文那一步
    result = await tool.run(
        event=object(),
        action="create",
        memory="用户偏好简短回复",
        memory_type="knowledge",
        reasoning="多次要求",
        tags=["偏好"],
    )

    assert "无法获取插件上下文" in result


@pytest.mark.asyncio
async def test_run_still_rejects_missing_judgment():
    """掰回形状后仍要按原规则拒绝空 judgment，不能因为容错就放过。"""
    tool = CoreMemoryRememberTool()

    result = await tool.run(
        event=object(),
        action="create",
        memory="",
        memory_type="knowledge",
        tags=["锚点"],
    )

    assert "memory.judgment 不能为空" in result


class _FakeRuntime:
    """记忆运行时替身：只记录 remember 收到的参数。"""

    def __init__(self):
        self.calls: list[dict] = []

    async def remember(self, **kwargs):
        self.calls.append(kwargs)
        return 42


class _FakePluginContext:
    def __init__(self, runtime):
        self._runtime = runtime

    def get_component(self, name):
        return self._runtime if name == "memory_runtime" else None

    async def resolve_memory_scope_from_event(self, event):
        return "test_scope"


class _FakeEvent:
    def __init__(self, plugin_context):
        self.plugin_context = plugin_context


@pytest.mark.asyncio
async def test_flattened_params_reach_remember_with_full_shape():
    """摊平的参数掰回后要能真正写入，四个字段一个不丢。"""
    runtime = _FakeRuntime()
    tool = CoreMemoryRememberTool()
    event = _FakeEvent(_FakePluginContext(runtime))

    result = await tool.run(
        event=event,
        action="create",
        memory="用户偏好简短回复",
        memory_type="knowledge",
        reasoning="多次要求",
        tags=["偏好", "回复风格"],
    )

    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert call["judgment"] == "用户偏好简短回复"
    assert call["memory_type"] == "knowledge"
    assert call["reasoning"] == "多次要求"
    assert call["tags"] == ["偏好", "回复风格"]
    assert call["memory_scope"] == "test_scope"
    assert "已记住" in result


@pytest.mark.asyncio
async def test_correctly_nested_params_also_reach_remember():
    """本来就传对的调用不受容错层影响，写入结果一致。"""
    runtime = _FakeRuntime()
    tool = CoreMemoryRememberTool()
    event = _FakeEvent(_FakePluginContext(runtime))

    result = await tool.run(
        event=event,
        action="create",
        memory={
            "judgment": "用户偏好简短回复",
            "memory_type": "knowledge",
            "reasoning": "多次要求",
            "tags": ["偏好"],
        },
    )

    assert len(runtime.calls) == 1
    assert runtime.calls[0]["judgment"] == "用户偏好简短回复"
    assert runtime.calls[0]["tags"] == ["偏好"]
    assert "已记住" in result
