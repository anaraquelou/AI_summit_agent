import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

def setup_logger(logger_name: str) -> logging.Logger:
    """Setup a logger with the standard configuration (RichHandler + FileHandler)."""
    target_logger = logging.getLogger(logger_name)
    target_logger.setLevel(logging.INFO)
    
    # Create new handlers for this logger (to avoid sharing handler state)
    shell_h = RichHandler()
    file_h = logging.FileHandler("debug.log")
    
    shell_h.setLevel(logging.INFO)
    file_h.setLevel(logging.INFO)

    # the formatter determines what our logs will look like
    fmt_shell = '%(message)s'
    fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'

    shell_formatter = logging.Formatter(fmt_shell)
    file_formatter = logging.Formatter(fmt_file)

    shell_h.setFormatter(shell_formatter)
    file_h.setFormatter(file_formatter)
    
    target_logger.addHandler(shell_h)
    target_logger.addHandler(file_h)
    target_logger.propagate = False
    
    return target_logger