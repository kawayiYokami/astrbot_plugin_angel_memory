"""
文件扫描服务

扫描raw文件夹中的文件，并自动同步到数据库。
"""

import os
import hashlib
import queue
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict

from astrbot.api import logger
from .timestamp_manager import TimestampManager
from ..llm_memory.service.note_service import NoteService


class FileMonitorService:
    """文件扫描服务类"""

    def __init__(self, data_directory: str, note_service: NoteService):
        """
        初始化文件扫描服务

        Args:
            data_directory: 插件数据目录
            note_service: 笔记服务实例
        """
        self.logger = logger
        self.data_directory = Path(data_directory)
        self.raw_directory = self.data_directory / "raw"
        self.note_service = note_service

        # 初始化时间戳管理器
        self.timestamp_manager = TimestampManager(data_directory)

        # 初始化任务队列和生产者-消费者控制
        self.task_queue = queue.Queue()
        self.all_tasks_submitted = threading.Event()
        self.shutdown_event = threading.Event()
        self.consumer_pool = None
        self.service_thread = None

        # 文件哈希缓存，避免重复处理
        self.file_hashes: Dict[str, str] = {}

        # 确保raw目录存在
        self.raw_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"文件扫描服务初始化完成，扫描目录: {self.raw_directory}")

    def start_monitoring(self):
        """启动一个后台守护线程来运行整个文件扫描服务"""
        try:
            self.service_thread = threading.Thread(target=self._run_service)
            self.service_thread.daemon = True  # 设置为守护线程
            self.service_thread.start()
            self.logger.info("文件扫描服务已在后台启动，不会阻塞初始化。")
        except Exception as e:
            self.logger.error(f"启动文件扫描后台服务失败: {e}")

    def stop_monitoring(self):
        """发送关闭信号，并等待后台服务优雅地停止"""
        try:
            self.logger.info("正在请求后台文件服务停止...")
            self.shutdown_event.set()  # 1. 通知消费者停止

            # 2. 清空队列，让 join() 可以快速返回
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                    self.task_queue.task_done()  # 必须调用，否则join会卡住
                except queue.Empty:
                    break

            # 3. 等待后台线程结束（最多5秒）
            if self.service_thread and self.service_thread.is_alive():
                self.service_thread.join(timeout=5)

            # 4. 关闭消费者线程池
            if self.consumer_pool:
                self.consumer_pool.shutdown(wait=True)

            # 5. 最后清理资源
            self.timestamp_manager.close()
            self.logger.info("文件扫描服务已成功停止。")
        except Exception as e:
            self.logger.error(f"停止文件扫描服务时发生错误: {e}")

    def _initial_scan(self):
        """首次扫描并将所有文件放入任务队列（生产者）"""
        try:
            self.logger.info("开始文件扫描...")

            # 收集所有文件并放入队列
            file_count = 0
            for file_path in self.raw_directory.rglob('*'):
                if file_path.is_file():
                    self.task_queue.put(str(file_path))
                    file_count += 1

            self.logger.info(f"文件扫描完成，共发现 {file_count} 个文件待处理")
            self.all_tasks_submitted.set()  # 通知消费者：没有更多任务了
        except Exception as e:
            self.logger.error(f"文件扫描失败: {e}")
            # 即使扫描失败，也要设置事件以防止消费者无限等待
            self.all_tasks_submitted.set()

    def _run_service(self):
        """在后台运行整个文件扫描和处理服务（守护线程）"""
        try:
            # 计算最优的工作线程数
            cpu_cores = os.cpu_count()
            if cpu_cores is None:
                num_workers = 1
                self.logger.info("无法检测到CPU核心数，将以单线程模式启动消费者。")
            else:
                # 使用所有核心数，但至少保证有1个线程
                num_workers = max(1, cpu_cores)
                self.logger.info(f"检测到 {cpu_cores} 个CPU核心，将以所有的核心数 ({num_workers}个) 启动消费者线程。")

            # 1. 启动消费者线程池
            self.consumer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
            for _ in range(num_workers):
                self.consumer_pool.submit(self._consumer_task)
            self.logger.info("消费者线程池已启动。")

            # 2. 启动生产者（直接调用，因为扫描很快）
            self._initial_scan()

            # 3. 等待所有任务处理完成
            self.task_queue.join()  # 阻塞，直到队列中所有任务都被 task_done()
            self.logger.info("所有文件处理任务已完成。")

        except Exception as e:
            self.logger.error(f"文件处理服务在后台运行时发生错误: {e}")
        finally:
            # 4. 安全关闭消费者线程池
            if self.consumer_pool:
                self.consumer_pool.shutdown(wait=True)
            self.logger.info("后台文件处理服务已停止。")

    def _consumer_task(self):
        """消费者任务：从队列中获取任务并处理，直到收到关闭信号"""
        while not self.shutdown_event.is_set():
            try:
                # 检查是否正常完成
                if self.all_tasks_submitted.is_set() and self.task_queue.empty():
                    break

                file_path = self.task_queue.get(timeout=1)  # 等待1秒
                if self.shutdown_event.is_set():  # 取出后再次检查关闭信号
                    break

                self._process_file(file_path)
                self.task_queue.task_done()
            except queue.Empty:
                continue  # 队列暂时为空，继续等待或检查关闭信号

    def _process_file(self, file_path: str):
        """
        处理单个文件

        Args:
            file_path: 文件路径
        """
        try:
            
            # 检查文件是否已更改，避免重复处理
            if not self.timestamp_manager.is_file_changed(file_path):
                return

            # 交给笔记服务处理文件（由服务层决定是否支持该文件类型）
            processed_count = self.note_service.parse_and_store_file(file_path)

            # 只要文件被成功处理（即使生成0个块），就更新时间戳以避免重复扫描
            self.timestamp_manager.update_file_timestamp(file_path)
            if processed_count > 0:
                self.logger.info(f"文件处理完成: {file_path}, 生成 {processed_count} 个文档块")
            else:
                self.logger.debug(f"文件处理完成: {file_path}, 未生成新的文档块（可能为空文件或内容无变化）")

        except Exception as e:
            # 一旦出现错误，记录日志并重新抛出异常，中断整个扫描过程
            self.logger.error(f"处理文件失败: {file_path}, 错误: {e}")
            raise  # 重新抛出异常以中断扫描过程

    def _remove_file_data(self, file_path: str):
        """
        移除文件相关的数据

        Args:
            file_path: 文件路径
        """
        try:
            # 从时间戳管理器中移除
            self.timestamp_manager.remove_file_timestamp(file_path)

            # 通过笔记服务删除与该文件相关的所有数据
            success = self.note_service.remove_file_data(file_path)

            if success:
                self.logger.info(f"文件已删除: {file_path}，相关数据已清理")
            else:
                self.logger.error(f"文件已删除: {file_path}，但数据清理失败")

        except Exception as e:
            self.logger.error(f"移除文件数据失败: {file_path}, 错误: {e}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """
        计算文件哈希值

        Args:
            file_path: 文件路径

        Returns:
            文件哈希值
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"计算文件哈希失败: {file_path}, 错误: {e}")
            return ""
