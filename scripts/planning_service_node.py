"""Read-only MoveIt clients with serialized, caller-driven ROS execution."""
import threading
import time

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


class PlanningServiceNode(Node):
    def __init__(self):
        self.planning_context = Context()
        rclpy.init(context=self.planning_context)
        super().__init__('bypass_policy_planning', context=self.planning_context)
        self.service_executor = SingleThreadedExecutor(context=self.planning_context)
        self.service_executor.add_node(self)
        self.service_lock = threading.Lock()
        self.service_closed = False
        self.request_count = 0
        self.response_timeout_count = 0

    def create_client(self, *args, **kwargs):
        client = super().create_client(*args, **kwargs)
        client._planning_request_and_wait = self.request_and_wait
        return client

    def request_and_wait(self, client, request, timeout):
        # Only this call spins this executor. No background spin thread, main
        # executor, sensor callback or action timer can consume its replies.
        deadline = time.monotonic() + timeout
        if not self.service_lock.acquire(timeout=timeout):
            raise TimeoutError('Another read-only planning request exceeded its budget')
        try:
            if getattr(self, 'service_closed', False):
                raise RuntimeError('Planning service worker has stopped')
            self.request_count += 1
            future = client.call_async(request)
            self.service_executor.spin_until_future_complete(
                future, timeout_sec=max(0., deadline-time.monotonic()))
            if not future.done():
                self.response_timeout_count += 1
            return future
        finally:
            self.service_lock.release()

    def close(self):
        with self.service_lock:
            self.service_closed = True
            self.service_executor.shutdown()
            self.destroy_node()
            rclpy.try_shutdown(context=self.planning_context)
