"""钉死消息内容的归一化口径：账本侧与上下文侧必须得到同一个纯文本。"""

from astrbot.core.agent.message import TextPart, ThinkPart
from astrbot_plugin_angel_memory.core.services.injection_service import (
    DeepMindInjectionService,
)

plain = DeepMindInjectionService._plain_text


def test_string_content_is_stripped():
    assert plain(" 今天几点睡 ") == "今天几点睡"


def test_dict_text_item_is_picked():
    assert plain([{"type": "text", "text": "今天几点睡"}]) == "今天几点睡"


def test_part_object_text_is_picked():
    assert plain([TextPart(text="今天几点睡")]) == "今天几点睡"


def test_dict_and_object_forms_agree():
    """同一条消息两种形态必须归一到同一个结果，否则配对会误判不一致。"""
    assert plain([TextPart(text="同一句话")]) == plain(
        [{"type": "text", "text": "同一句话"}]
    )


def test_mixed_forms_are_joined_in_order():
    content = ["开场", {"type": "text", "text": "中间"}, TextPart(text="结尾")]

    assert plain(content) == "开场 中间 结尾"


def test_non_text_parts_are_skipped():
    content = [
        {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}},
        ThinkPart(think="想一想"),
    ]

    assert plain(content) == ""


def test_empty_and_unsupported_content_falls_back_to_empty_string():
    assert plain([]) == ""
    assert plain("") == ""
    assert plain(None) == ""
    assert plain(123) == ""
    assert plain([TextPart(text="")]) == ""


def test_bare_dict_outside_list_is_not_treated_as_content():
    """content 只在 str 与 list 两种形态上有口径，单个 dict 不属于其中。"""
    assert plain({"type": "text", "text": "不在列表里"}) == ""
