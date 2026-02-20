import time
from .sleep_maintenance_service import SleepMaintenanceService


class DeepMindSleepService:
    """DeepMind 的睡眠巩固职责。"""

    def __init__(self, deepmind):
        self.deepmind = deepmind
        self.maintenance_service = SleepMaintenanceService(deepmind)

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
                deepmind.logger.info("[睡眠] 阶段=前置维护 开始")
                await self.maintenance_service.run_pre_consolidate()
                deepmind.logger.info("[睡眠] 阶段=前置维护 完成")

                deepmind.logger.info("[睡眠] 阶段=记忆清理 开始")
                await deepmind.memory_system.consolidate_memories()
                deepmind.logger.info("[睡眠] 阶段=记忆清理 完成")
                cleanup_completed_at = time.time()
                deepmind.logger.info("[睡眠] 阶段=后置维护 开始")
                await self.maintenance_service.run_post_consolidate(
                    cleanup_completed_at=cleanup_completed_at
                )
                deepmind.logger.info("[睡眠] 阶段=后置维护 完成")
                return True
            else:
                deepmind.logger.error("记忆系统不可用，跳过巩固")
                return False
        except Exception as e:
            elapsed_time = time.time() - start_time
            deepmind.logger.error(f"记忆巩固失败（耗时 {elapsed_time:.2f} 秒）: {e}")
            return False
