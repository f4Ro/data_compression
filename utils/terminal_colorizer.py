from typing import Any


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printc(*args: Any, color: str = 'green') -> None:
    string = ''
    for arg in args:
        string += str(arg)
    print(_get_color_from_string(color) + string + bcolors.ENDC)


def _get_color_from_string(string: str) -> str:
    return {
        'green': bcolors.OKGREEN,
        'blue': bcolors.OKBLUE,
        'cyan': bcolors.OKCYAN,
        'yellow': bcolors.WARNING,
        'red': bcolors.FAIL
    }[string.lower()]
