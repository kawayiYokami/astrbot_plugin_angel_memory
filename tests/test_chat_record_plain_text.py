"""对话记录拼接：原文照搬、按时间排序、过滤空内容与工具调用、收集参与者。

天使之心模式下 prompt 已带「[名字 (ID: 号)] (时间): 」前缀，原生模式下
prompt 就是原文；两种都原样照搬，不补署名行、不缩进、不插时间行、
不渲染引用。
"""

from astrbot_plugin_angel_memory.core.utils.small_model_prompt_builder import (
    SmallModelPromptBuilder,
)


def user(name: str, sid: str, content, when: float) -> dict:
    return {
        "role": "user",
        "sender_name": name,
        "sender_id": sid,
        "content": content,
        "timestamp": when,
    }


def assistant(content, when: float) -> dict:
    return {
        "role": "assistant",
        "sender_name": "助理",
        "sender_id": "assistant",
        "content": content,
        "timestamp": when,
    }


def fmt(records) -> str:
    text, _ = SmallModelPromptBuilder.format_chat_records(records)
    return text


# ---------- 一、原文照搬 ----------


def test_angelheart_prefix_is_kept_verbatim():
    """天使之心加的前缀原样保留，不再叠一层署名。"""
    raw = "[红豆泥 (ID: 289104862)] (2026-11-07 15:09): 机械键盘的客制化是什么意思？"

    assert fmt([user("红豆泥", "289104862", raw, 1000.0)]) == raw


def test_group_prefix_with_role_is_kept_verbatim():
    """群聊形态的「[群友: 名字 (ID: 号, 角色)]」同样原样保留。"""
    raw = "[群友: 红豆泥 (ID: 289104862, 群友)] (2026-11-07 15:09): 在吗"

    assert fmt([user("红豆泥", "289104862", raw, 1000.0)]) == raw


def test_no_speaker_header_is_added():
    """没有前缀的原文不会被补上名字（号）。"""
    assert fmt([user("红豆泥", "289104862", "在吗", 1000.0)]) == "在吗"


def test_assistant_content_has_no_prefix():
    """助理发言不加任何前缀。"""
    assert fmt([assistant("在的，一直都在。", 1000.0)]) == "在的，一直都在。"


def test_multiline_content_is_kept_as_is():
    """多行内容不缩进、不重排。"""
    raw = "第一行\n第二行\n\n第四行"

    assert fmt([user("甲", "1", raw, 1000.0)]) == raw


def test_quote_fragment_is_not_rendered():
    """引用片段保持原文，不转成「> 」引用块。"""
    raw = "[引用消息(暗橡: 不应该把代码拿过来分发)] fairy 解释一下为什么不应该"

    assert fmt([user("甲", "1", raw, 1000.0)]) == raw


def test_no_clock_line_is_inserted():
    """不再插时间行，间隔多久都不插。"""
    text = fmt(
        [
            user("甲", "1", "早", 1000.0),
            user("甲", "1", "晚", 1000.0 + 3600 * 5),
        ]
    )

    assert text == "早\n晚"


def test_no_markdown_decorations_added():
    """不额外加加粗、标题、代码块这类修饰。"""
    text = fmt([user("甲", "1", "普通一句话", 1000.0)])

    assert text == "普通一句话"


# ---------- 二、排序 ----------


def test_records_are_sorted_by_timestamp():
    text = fmt(
        [
            user("甲", "1", "第二句", 2000.0),
            user("甲", "1", "第一句", 1000.0),
        ]
    )

    assert text == "第一句\n第二句"


def test_same_timestamp_keeps_input_order():
    text = fmt(
        [
            user("甲", "1", "先说的", 1000.0),
            user("乙", "2", "后说的", 1000.0),
        ]
    )

    assert text == "先说的\n后说的"


# ---------- 三、过滤 ----------


def test_empty_content_is_skipped():
    text = fmt(
        [
            user("甲", "1", "", 1000.0),
            user("甲", "1", "  ", 1001.0),
            user("甲", "1", "有内容", 1002.0),
        ]
    )

    assert text == "有内容"


def test_structured_toolcall_is_skipped():
    record = user("甲", "1", "不该出现", 1000.0)
    record["is_structured_toolcall"] = True

    assert fmt([record]) == ""


def test_toolcall_keyword_content_is_skipped():
    text = fmt(
        [
            user("甲", "1", "工具调用结果：成功", 1000.0),
            user("甲", "1", "正常发言", 1001.0),
        ]
    )

    assert text == "正常发言"


# ---------- 四、参与者 ----------


def test_participants_are_deduped_by_sender_id():
    _, users = SmallModelPromptBuilder.format_chat_records(
        [
            user("甲", "1", "一", 1000.0),
            user("甲", "1", "二", 1001.0),
            user("乙", "2", "三", 1002.0),
        ]
    )

    assert [item["id"] for item in users] == ["1", "2"]


def test_assistant_is_not_a_participant():
    _, users = SmallModelPromptBuilder.format_chat_records(
        [
            user("甲", "1", "问", 1000.0),
            assistant("答", 1001.0),
        ]
    )

    assert [item["id"] for item in users] == ["1"]


def test_unknown_sender_id_is_not_collected():
    _, users = SmallModelPromptBuilder.format_chat_records([user("未知", "unknown", "在吗", 1000.0)])

    assert users == []


def test_missing_sender_name_falls_back_to_group_label():
    _, users = SmallModelPromptBuilder.format_chat_records([user("", "42", "在吗", 1000.0)])

    assert users == [{"id": "42", "name": "群友"}]
