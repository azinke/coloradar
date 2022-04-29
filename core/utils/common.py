"""Common.

Collection of general usage functions
"""
from .color import Color


def error(msg: str) -> None:
    """Print an error message.

    The message would be printed in red in the terminal

    Argument:
        msg: Error message to print
    """
    print(Color.RED, "[ERROR] " + msg, Color.RESET)

def warning(msg: str) -> None:
    """Print a warning message.

    The message would be printed in yellow in the terminal

    Argument:
        msg: Error message to print
    """
    print(Color.YELLOW, "[WARNING] " + msg, Color.RESET)

def info(msg: str) -> None:
    """Print a warning message.

    The message would be printed in yellow in the terminal

    Argument:
        msg: Error message to print
    """
    print(Color.RESET, "[INFO] " + msg)

def success(msg: str) -> None:
    """Print a success message.

    The message would be printed in green in the terminal

    Argument:
        msg: Error message to print
    """
    print(Color.GREEN, "[SUCCESS] " + msg, Color.RESET)
