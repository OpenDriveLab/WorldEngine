import logging
from mmcv.utils import get_logger as get_real_logger

class LazyLogger:
    """
    A proxy for the real logger. It delays the logger initialization
    until the first time a logging method is called.
    
    This allows defining loggers at the module level in distributed training
    scenarios, where the logger's behavior (especially its rank-specific
    level) depends on a runtime state (dist.is_initialized()) that is
    not available at import time.
    """
    def __init__(self, name, log_file=None, log_level=logging.INFO, file_mode='w'):
        self.name = 'mmdet.' + name
        self.log_file = log_file
        self.log_level = log_level
        self.file_mode = file_mode
        self._logger = None

    def _get_logger(self):
        """Initialize and cache the real logger instance."""
        if self._logger is None:
            # This is the moment of truth: get_real_logger is called here,
            # when dist is already initialized.
            self._logger = get_real_logger(
                name=self.name,
                log_file=self.log_file,
                log_level=self.log_level,
                file_mode=self.file_mode
            )
        return self._logger

    def __getattr__(self, name):
        """
        Forward any attribute access (e.g., .info, .warning)
        to the real logger.
        """
        # The first time you call logger.info, logger.warning, etc.
        # this method will be triggered.
        return getattr(self._get_logger(), name)

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    return LazyLogger(name, log_file, log_level, file_mode)
