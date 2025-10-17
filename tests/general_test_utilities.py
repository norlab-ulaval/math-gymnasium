# coding=utf-8

from tools.console_tools.format import ConsoleFormat


def numpy_array_print_precision_warning(precision: int) -> None:
    print(
        f"\n{ConsoleFormat.MSG_WARNING_FORMAT}WARNING › numpy array printing with "
        f"{precision} decimal precision{ConsoleFormat.MSG_END_FORMAT}"
    )
    return None
