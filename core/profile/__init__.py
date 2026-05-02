"""
画像模块 — 跨群聊用户事实画像
提供画像提取器和注入器，实现 user_profile scope。
"""
from .config import ProfileExtractionConfig
from .extractor import ProfileExtractor
from .injector import ProfileInjector
