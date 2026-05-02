"""
画像模块 — 配置文件测试
"""
import sys
import os
import importlib

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base)

# 绕过 core/__init__.py 的直接导入链
spec = importlib.util.spec_from_file_location(
    "profile_config",
    os.path.join(base, "core", "profile", "config.py"),
)
config_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_mod)
ProfileExtractionConfig = config_mod.ProfileExtractionConfig


class TestGreenTags:
    def test_default_green_tags_not_empty(self):
        config = ProfileExtractionConfig()
        assert len(config.green_tags) >= 5

    def test_all_green_tags_are_strings(self):
        config = ProfileExtractionConfig()
        for tag in config.green_tags:
            assert isinstance(tag, str)

    def test_is_green_recognizes_members(self):
        config = ProfileExtractionConfig()
        assert config.is_green("preferred_name") is True
        assert config.is_green("not_a_tag") is False


class TestRedKeywords:
    def test_password_blocked(self):
        config = ProfileExtractionConfig()
        assert config.contains_red("我的密码是abc") is True

    def test_token_blocked(self):
        config = ProfileExtractionConfig()
        assert config.contains_red("这是api_key: xyz") is True

    def test_secret_blocked(self):
        config = ProfileExtractionConfig()
        assert config.contains_red("别跟别人说") is True

    def test_normal_text_passes(self):
        config = ProfileExtractionConfig()
        assert config.contains_red("我叫小貔貅") is False

    def test_phone_blocked(self):
        config = ProfileExtractionConfig()
        assert config.contains_red("我的电话是138") is True


class TestYellowTags:
    def test_default_yellow_disabled(self):
        config = ProfileExtractionConfig()
        assert config.is_yellow_enabled("personal_experience") is False

    def test_yellow_enabled_when_added(self):
        config = ProfileExtractionConfig()
        config.yellow_tags.append("personal_experience")
        assert config.is_yellow_enabled("personal_experience") is True


class TestDefaults:
    def test_extraction_interval_positive(self):
        assert ProfileExtractionConfig().extraction_interval > 0

    def test_max_profile_items_reasonable(self):
        assert 10 <= ProfileExtractionConfig().max_profile_items <= 50

    def test_dedup_threshold_in_range(self):
        assert 0.5 <= ProfileExtractionConfig().dedup_threshold <= 1.0
