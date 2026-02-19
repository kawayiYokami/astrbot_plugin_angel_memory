import time
from .simple_memory_backup_service import SimpleMemoryBackupService


class DeepMindSleepService:
    """DeepMind 的睡眠巩固职责。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind

    async def check_and_sleep_if_needed(self, sleep_interval: int) -> bool:
        deepmind = self.deepmind

        if sleep_interval <= 0:
            return False

        current_time = time.time()
        if deepmind.last_sleep_time is None:
            sleep_success = await self.sleep()
            if sleep_success:
                deepmind.last_sleep_time = current_time
            return sleep_success

        time_since_last_sleep = current_time - deepmind.last_sleep_time
        if time_since_last_sleep >= sleep_interval:
            sleep_success = await self.sleep()
            if sleep_success:
                deepmind.last_sleep_time = current_time
            return sleep_success

        return False

    async def sleep(self) -> bool:
        deepmind = self.deepmind
        if not deepmind.is_enabled():
            return False

        start_time = time.time()
        try:
            if deepmind.memory_system is not None:
                deepmind.logger.info("[sleep] phase=consolidate start")
                await deepmind.memory_system.consolidate_memories()
                deepmind.logger.info("[sleep] phase=consolidate done")
                deepmind.logger.info("[sleep] phase=backup start")
                await self._backup_vector_memories_after_sleep()
                deepmind.logger.info("[sleep] phase=backup done")
                return True
            else:
                deepmind.logger.error("记忆系统不可用，跳过巩固")
                return False
        except Exception as e:
            elapsed_time = time.time() - start_time
            deepmind.logger.error(f"记忆巩固失败（耗时 {elapsed_time:.2f} 秒）: {e}")
            return False

    async def _backup_vector_memories_after_sleep(self) -> None:
        deepmind = self.deepmind
        if bool(getattr(deepmind.config, "enable_simple_memory", False)):
            return

        try:
            plugin_context = getattr(deepmind, "plugin_context", None)
            if plugin_context is None:
                return

            cognitive_service = plugin_context.get_component("cognitive_service")
            memory_sql_manager = plugin_context.get_component("memory_sql_manager")
            if not cognitive_service or not hasattr(cognitive_service, "main_collection"):
                deepmind.logger.debug("[simple_backup] skip: cognitive_service/main_collection 不可用")
                return
            if memory_sql_manager is None:
                deepmind.logger.warning("[simple_backup] skip: memory_sql_manager 不可用")
                return

            backup_service = SimpleMemoryBackupService(deepmind.logger)
            await backup_service.backup_from_collection(
                collection=cognitive_service.main_collection,
                memory_sql_manager=memory_sql_manager,
                source="sleep",
                provider_id=str(plugin_context.get_current_provider()),
            )
        except Exception as e:
            deepmind.logger.error(f"[simple_backup] failed source=sleep error={e}", exc_info=True)
