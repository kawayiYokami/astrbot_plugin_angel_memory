"""剧本式对话记录排版的断言。

覆盖：分段与署名、时间断档、时间排序、引用单列（含多行与半角括号）、@ 占位符、署名兜底。
"""

from datetime import datetime, timedelta, timezone

from astrbot_plugin_angel_memory.core.utils.small_model_prompt_builder import (
    SmallModelPromptBuilder,
)

TZ = timezone(timedelta(hours=8))


def ts(hour: int, minute: int, second: int = 0) -> float:
    """构造当天 UTC+8 的时间戳。"""
    return datetime(2026, 11, 7, hour, minute, second, tzinfo=TZ).timestamp()


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


# ---------- 一、换人分段 ----------

def test_alternating_speakers_get_one_header_each():
    """跨人交替发言：每人一段，署名只出现一次。"""
    text = fmt(
        [
            user("甲", "1", "你好", ts(12, 0)),
            user("乙", "2", "在", ts(12, 1)),
            user("甲", "1", "有事吗", ts(12, 2)),
        ]
    )
    assert text == "12:00\n甲（1）\n  你好\n乙（2）\n  在\n甲（1）\n  有事吗"
    assert text.count("甲（1）") == 2, "换人后再次发言应当重新起段署名"
    assert text.splitlines().count("乙（2）") == 1


def test_same_speaker_consecutive_messages_stay_in_one_section():
    """同一人连续三条：一段内三行，不重复署名。"""
    text = fmt(
        [
            user("甲", "1", "一", ts(12, 0)),
            user("甲", "1", "二", ts(12, 0, 30)),
            user("甲", "1", "三", ts(12, 1)),
        ]
    )
    assert text == "12:00\n甲（1）\n  一\n  二\n  三"
    assert text.count("甲（1）") == 1


# ---------- 二、时间断档 ----------

def test_gap_under_ten_minutes_does_not_insert_clock():
    """9 分钟断档不插时间行。"""
    text = fmt(
        [
            user("甲", "1", "第一条", ts(12, 0)),
            user("甲", "1", "九分钟后", ts(12, 9)),
        ]
    )
    assert text.count("12:00") == 1
    assert "12:09" not in text


def test_gap_over_ten_minutes_inserts_clock():
    """11 分钟断档插时间行。"""
    text = fmt(
        [
            user("甲", "1", "第一条", ts(12, 0)),
            user("甲", "1", "十一分钟后", ts(12, 11)),
        ]
    )
    assert "12:00" in text
    assert "12:11" in text
    assert text.splitlines().index("12:11") < text.splitlines().index("  十一分钟后")


def test_first_message_always_prefixed_with_clock():
    """第一条消息前面必须插时间。"""
    text = fmt([user("甲", "1", "开头", ts(9, 5))])
    assert text.splitlines()[0] == "09:05"


def test_clock_uses_utc_plus_eight():
    """时间按 UTC+8 输出：UTC 04:00 对应北京时间 12:00。"""
    utc_timestamp = datetime(2026, 11, 7, 4, 0, tzinfo=timezone.utc).timestamp()
    text = fmt([user("甲", "1", "内容", utc_timestamp)])
    assert text.splitlines()[0] == "12:00"


def test_gap_forces_repeated_header_for_same_speaker():
    """插了时间行之后，同一个人再说话也要重新署名。"""
    text = fmt(
        [
            user("甲", "1", "先说", ts(12, 40)),
            user("甲", "1", "隔了二十多分钟", ts(13, 5)),
        ]
    )
    assert text.splitlines() == [
        "12:40",
        "甲（1）",
        "  先说",
        "13:05",
        "甲（1）",
        "  隔了二十多分钟",
    ]
    assert text.count("甲（1）") == 2, "时间行之后必须重新署名"


def test_short_gap_keeps_same_speaker_without_repeated_header():
    """断档不到十分钟不插时间行，同一个人继续接下去，不重复署名。"""
    text = fmt(
        [
            user("甲", "1", "先说", ts(12, 0)),
            user("甲", "1", "八分钟后", ts(12, 8)),
            user("甲", "1", "九分钟后", ts(12, 9)),
        ]
    )
    assert "12:08" not in text and "12:09" not in text
    assert text.count("甲（1）") == 1, "没有时间行就接在同一段里"
    assert text.splitlines() == ["12:00", "甲（1）", "  先说", "  八分钟后", "  九分钟后"]


def test_exactly_six_hundred_seconds_inserts_clock():
    """正好 600 秒要插时间行（判定是 >= 而不是 >），同一人也重新署名。"""
    text = fmt(
        [
            user("甲", "1", "整点前", ts(12, 0)),
            user("甲", "1", "整点后", ts(12, 10)),
        ]
    )
    assert text.splitlines() == [
        "12:00",
        "甲（1）",
        "  整点前",
        "12:10",
        "甲（1）",
        "  整点后",
    ]


# ---------- 三、时间排序 ----------

def test_out_of_order_input_is_sorted_by_timestamp():
    """打乱顺序的输入按 timestamp 升序输出。"""
    text = fmt(
        [
            user("乙", "2", "后说的", ts(12, 5)),
            user("甲", "1", "先说的", ts(12, 0)),
        ]
    )
    assert text.index("先说的") < text.index("后说的")
    assert text.splitlines()[0] == "12:00"


def test_same_timestamp_keeps_input_order():
    """同一时间戳的两条保持传入顺序。"""
    text = fmt(
        [
            user("先", "1", "AAA", ts(12, 0)),
            user("后", "2", "BBB", ts(12, 0)),
        ]
    )
    assert text.index("AAA") < text.index("BBB")


# ---------- 四、引用单列 ----------

def test_quote_fragment_becomes_reference_line():
    """引用片段单列成 > 名：内容，且原正文不丢。"""
    text = fmt([user("甲", "1", "[引用消息(乙: 原文内容)]我的回复", ts(12, 0))])
    lines = text.splitlines()
    assert lines[0] == "12:00"
    assert lines[1] == "甲（1）"
    assert lines[2] == "  > 乙：原文内容"
    assert lines[3] == "  我的回复"
    assert "引用消息" not in text


def test_quote_with_nested_brackets_keeps_outer_boundary():
    """引用里嵌套圆括号与 [At:...] 时，要吃到最外层右括号。"""
    text = fmt(
        [user("甲", "1", "[引用消息(乙: 看这个(重要) [At:99])]好的", ts(12, 0))]
    )
    assert "  > 乙：看这个(重要) [At:99]" in text
    assert "  好的" in text
    assert "引用消息" not in text


def test_message_with_only_quote_still_renders():
    """只剩引用、正文为空的消息，仍输出引用行。"""
    text = fmt([user("甲", "1", "[引用消息(乙: 只有引用)]", ts(12, 0))])
    assert "  > 乙：只有引用" in text
    assert "甲（1）" in text


def test_multiline_quote_prefixes_every_line():
    """被引内容带换行时，每一行都要带「  > 」前缀，不能出现顶格续行。"""
    text = fmt([user("甲", "1", "[引用消息(乙: 第一行\n第二行)]好的", ts(12, 0))])
    assert text.splitlines() == [
        "12:00",
        "甲（1）",
        "  > 乙：第一行",
        "  > 第二行",
        "  好的",
    ]
    # 顶格行只允许是时间行或发言人署名行，续行必须缩进
    assert "\n第二行" not in text
    for line in text.splitlines():
        if line and not line.startswith(" "):
            assert line in ("12:00", "甲（1）"), f"顶格行只能是时间或署名：{line!r}"


def test_quote_with_halfwidth_paren_is_not_truncated():
    """被引内容含半角 ) 时不能被截断，")]" 才是收尾。"""
    cases = [
        ("[引用消息(乙: 今天天气:) 挺好)]我的回复", "  > 乙：今天天气:) 挺好"),
        ("[引用消息(乙: 第一条1) 第二条)]我的回复", "  > 乙：第一条1) 第二条"),
        ("[引用消息(乙: 哈哈哈(笑))]我的回复", "  > 乙：哈哈哈(笑)"),
    ]
    for content, expected_quote in cases:
        text = fmt([user("甲", "1", content, ts(12, 0))])
        assert expected_quote in text, f"引用被截断了：{content!r} -> {text!r}"
        assert "  我的回复" in text
        assert "引用消息" not in text


# ---------- 五、@ 与占位符留在正文 ----------

def test_at_placeholder_stays_in_body_and_is_not_treated_as_speaker():
    """[At:...] 是正文的一部分，不能被当发言人或署名。"""
    text = fmt([user("甲", "1", "[At:123456] 你看这个", ts(12, 0))])
    lines = text.splitlines()
    assert lines[1] == "甲（1）"
    assert lines[2] == "  [At:123456] 你看这个"
    assert "123456（" not in text


def test_placeholders_are_preserved_verbatim():
    """[图片]、[表情:x] 这类占位符原样保留。"""
    text = fmt([user("甲", "1", "[图片]", ts(12, 0))])
    assert "  [图片]" in text

    text = fmt([user("甲", "1", "他说[表情:微笑]好的", ts(12, 1))])
    assert " 他说[表情:微笑]好的".strip() in text


# ---------- 六、署名兜底与助手段 ----------

def test_missing_name_falls_back_to_group_and_id():
    """sender_name 缺失时用 群友（sender_id）兜底。"""
    text = fmt([user("", "2664", "内容", ts(12, 0))])
    assert "群友（2664）" in text


def test_missing_name_and_id_falls_back_to_group_only():
    """名字与号都缺失时只写 群友。"""
    text = fmt([user("", "", "内容", ts(12, 0))])
    assert text.splitlines()[1] == "群友"


def test_assistant_gets_its_own_section_labelled_wo():
    """assistant 消息段首写 我，不与群友混成一段。"""
    text = fmt(
        [
            user("甲", "1", "问题", ts(12, 0)),
            assistant("回答", ts(12, 1)),
            user("甲", "1", "追问", ts(12, 2)),
        ]
    )
    lines = text.splitlines()
    assert lines == ["12:00", "甲（1）", "  问题", "我", "  回答", "甲（1）", "  追问"]
    assert "[助理]" not in text


def test_empty_content_is_skipped():
    """内容为空的记录直接跳过，不产出空段。"""
    text = fmt(
        [
            user("甲", "1", "", ts(12, 0)),
            user("乙", "2", "有内容", ts(12, 1)),
        ]
    )
    assert "甲（1）" not in text
    assert "乙（2）" in text


# ---------- 七、参与者清单 ----------

def test_participants_list_deduped_by_sender_id():
    """参与者清单按发送者去重，id 与 name 保持原结构。"""
    _, users = SmallModelPromptBuilder.format_chat_records(
        [
            user("甲", "1", "一", ts(12, 0)),
            user("甲", "1", "二", ts(12, 1)),
            user("乙", "2", "三", ts(12, 2)),
        ]
    )
    assert users == [
        {"id": "1", "name": "甲"},
        {"id": "2", "name": "乙"},
    ]


# ---------- 八、不使用 markdown 修饰 ----------

def test_output_uses_no_markdown_decorations():
    """除了「  > 」这个引用排版符号，不出现加粗、标题、代码块等 markdown 修饰。"""
    text = fmt(
        [
            user("甲", "1", "普通内容", ts(12, 0)),
            assistant("回答", ts(12, 1)),
            user("乙", "2", "[引用消息(甲: 被引内容)]回复", ts(12, 2)),
        ]
    )
    for marker in ("**", "__", "##", "###", "`", "```", "~~", "- ["):
        assert marker not in text, f"出现了 markdown 修饰：{marker!r}"
    # 顶格的 ">" 是 markdown 引用块，出现即视为修饰越界
    assert not [line for line in text.splitlines() if line.startswith(">")]
    # 「  > 」是引用行的排版符号，不属于被禁的 markdown 修饰，按既定样例保留
    assert text.count("  > ") == 1
    assert [line for line in text.splitlines() if line.startswith("  > ")] == [
        "  > 甲：被引内容"
    ]
