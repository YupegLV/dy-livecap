"""异步任务管理

管理直播录制任务，在独立线程中执行，完成后回调通知。
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    url: str
    duration: int  # 分钟
    user_openid: str
    status: TaskStatus = TaskStatus.PENDING
    result: dict = field(default_factory=dict)
    error: str = ""
    progress: str = ""
    output_dir: str = ""
    user_request: str = ""
    chat_history: list = field(default_factory=list)  # 多轮对话历史


class TaskManager:
    """异步录制任务管理器。"""

    def __init__(self, on_complete: Callable[[Task], None] | None = None):
        self._tasks: dict[str, Task] = {}
        self._on_complete = on_complete
        self._lock = threading.Lock()

    def submit(self, url: str, duration: int, user_openid: str,
               run_fn: Callable[[Task], dict], user_request: str = "") -> str:
        """提交一个录制任务。

        Args:
            url: 抖音分享链接
            duration: 录制时长（分钟）
            user_openid: 用户 QQ openid
            run_fn: 执行函数，接收 Task，返回结果 dict

        Returns:
            task_id
        """
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        task = Task(
            id=task_id,
            url=url,
            duration=duration,
            user_openid=user_openid,
            user_request=user_request,
        )

        with self._lock:
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._run_task,
            args=(task, run_fn),
            daemon=True,
        )
        thread.start()
        return task_id

    def get_task(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_user_tasks(self, user_openid: str) -> list[Task]:
        """获取用户的所有任务。"""
        with self._lock:
            return [
                t for t in self._tasks.values()
                if t.user_openid == user_openid
            ]

    def get_user_latest_completed_task(self, user_openid: str) -> Task | None:
        """获取用户最近完成的任务（用于多轮对话上下文）。"""
        with self._lock:
            completed = [
                t for t in self._tasks.values()
                if t.user_openid == user_openid and t.status == TaskStatus.COMPLETED
            ]
            if not completed:
                return None
            # 按任务 ID（时间戳）降序，返回最新的
            completed.sort(key=lambda t: t.id, reverse=True)
            return completed[0]

    def recover(self, task_id: str, run_fn: Callable[[Task], dict]) -> Task | None:
        """恢复失败的任务，从后续步骤重新执行。

        Args:
            task_id: 要恢复的任务 ID
            run_fn: 恢复执行函数，接收 Task，返回结果 dict

        Returns:
            恢复的 Task，如果 task_id 不存在或任务未失败则返回 None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

        # 重置任务状态
        task.status = TaskStatus.RUNNING
        task.progress = "正在恢复..."
        task.error = ""
        task.result = {}

        thread = threading.Thread(
            target=self._run_task,
            args=(task, run_fn),
            daemon=True,
        )
        thread.start()
        return task

    def _run_task(self, task: Task, run_fn: Callable[[Task], dict]):
        """在独立线程中执行任务。"""
        try:
            task.status = TaskStatus.RUNNING
            task.progress = "正在录制..."
            result = run_fn(task)
            task.status = TaskStatus.COMPLETED
            task.result = result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            print(f"[TaskManager] 任务 {task.id} 失败: {e}")

        if self._on_complete:
            try:
                self._on_complete(task)
            except Exception as e:
                print(f"[TaskManager] 回调失败: {e}")
