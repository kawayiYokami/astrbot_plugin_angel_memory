"""
WebUI 后端 API 模块

为 Plugin Pages 提供数据接口，通过 AstrBot 的 register_web_api 机制注册。
"""

from .routes import register_all_routes

__all__ = ["register_all_routes"]
