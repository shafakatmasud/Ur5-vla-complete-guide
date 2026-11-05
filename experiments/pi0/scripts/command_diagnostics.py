"""Queue command-path diagnostics; ROS/terminal logging runs on the main node."""
from queue import SimpleQueue, Empty


class QueuedCommandLogger:
    def __init__(self):
        self.messages = SimpleQueue()

    def info(self, message):
        self.messages.put(('info', str(message)))

    def warning(self, message):
        self.messages.put(('warning', str(message)))

    def error(self, message):
        self.messages.put(('error', str(message)))

    def flush(self, logger):
        # No command callback invokes this method. Preserve diagnostics rather
        # than dropping them if terminal/rosout logging is temporarily slow.
        for _ in range(16):
            try:
                level, message = self.messages.get_nowait()
            except Empty:
                break
            # rclpy caches severity by caller location. Each severity needs a
            # separate call site, otherwise INFO followed by ERROR raises.
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            else:
                logger.error(message)
