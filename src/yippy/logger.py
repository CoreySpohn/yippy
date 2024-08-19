"""Logging module."""

import logging

lib_name = "yippy"
# See https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 for
# info on the color codes
lib_color = "229"


# ANSI escape sequences for colors
class ColorCodes:
    """ANSI escape sequences for colors."""

    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    LIB = f"\033[38;5;{lib_color}m"


# Custom formatter to add colors
class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""

    COLORS = {
        logging.DEBUG: ColorCodes.BLUE,
        logging.INFO: ColorCodes.GREEN,
        logging.WARNING: ColorCodes.YELLOW,
        logging.ERROR: ColorCodes.RED,
        logging.CRITICAL: ColorCodes.MAGENTA,
    }

    def format(self, record: logging.LogRecord):
        """Format the log message with colors."""
        log = super().format(record)
        color = self.COLORS.get(record.levelno, ColorCodes.WHITE)
        return f"{ColorCodes.LIB}\033[48;5;16m[{lib_name}]\033[0m {color}{log}"


logger = logging.getLogger(f"{lib_name}")

shell_handler = logging.StreamHandler()
file_handler = logging.FileHandler("debug.log")

logger.setLevel(logging.DEBUG)
shell_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

shell_fmt = "%(levelname)s [%(asctime)s] \033[0m%(message)s"
file_fmt = (
    f"[{lib_name}] %(levelname)s %(asctime)s [%(filename)s:"
    "%(funcName)s:%(lineno)d] %(message)s"
)
shell_formatter = ColorFormatter(shell_fmt)
file_formatter = logging.Formatter(file_fmt)

shell_handler.setFormatter(shell_formatter)
file_handler.setFormatter(file_formatter)

logger.addHandler(shell_handler)
logger.addHandler(file_handler)

logger.propagate = True
