import os
import sys


def _supports_color(stream) -> bool:
    try:
        if os.environ.get("NO_COLOR") is not None:
            return False
        if hasattr(stream, "isatty") and stream.isatty():
            return True
    except Exception:
        pass
    return False


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    ORANGE = "\033[38;5;208m"


_ENABLE_COLOR = _supports_color(sys.stdout)


def _color(text: str, color_code: str) -> str:
    if not _ENABLE_COLOR:
        return text
    return f"{color_code}{text}{_Ansi.RESET}"


def format_full_line_if_tagged(line: str) -> str:
    """If line starts with '[UI]' or '[CORE]', color the entire line.

    - '[UI]'   -> yellow whole line
    - '[CORE]' -> blue whole line
    """
    if not isinstance(line, str):
        try:
            line = str(line)
        except Exception:
            return line

    if line.startswith("[ERROR]"):
        return _color(line, _Ansi.RED)
    if line.startswith("[WARN]"):
        # Fallback to YELLOW if 256-color not supported; our _color will just pass through if disabled
        return _color(line, _Ansi.ORANGE)
    if line.startswith("[UI]"):
        return _color(line, _Ansi.YELLOW)
    if line.startswith("[CORE]"):
        return _color(line, _Ansi.BLUE)
    return line


def install_print_colorizer() -> None:
    """Monkey-patch builtins.print to colorize tagged lines.

    Keeps behavior identical to print(), only transforms when the line begins with
    one of: [ERROR], [WARN], [UI], [CORE].
    """
    import builtins
    # Enable ANSI on Windows terminals (via colorama) when possible
    try:
        import colorama  # type: ignore
        try:
            # Automatically enables VT processing on Windows 10+ PowerShell/CMD
            colorama.just_fix_windows_console()
        except Exception:
            try:
                colorama.init(convert=True, strip=False)
            except Exception:
                pass
    except Exception:
        pass

    original_print = builtins.print

    def color_print(*args, **kwargs):
        try:
            if not args:
                return original_print(*args, **kwargs)

            first = args[0]
            if isinstance(first, str) and (
                first.startswith("[ERROR]")
                or first.startswith("[WARN]")
                or first.startswith("[UI]")
                or first.startswith("[CORE]")
            ):
                sep = kwargs.get("sep", " ")
                try:
                    line = sep.join(str(a) for a in args)
                except Exception:
                    # Fallback to original behavior if join fails
                    return original_print(*args, **kwargs)
                colored = format_full_line_if_tagged(line)
                # Preserve end/file/flush, but ignore sep since we pre-joined
                end = kwargs.get("end", "\n")
                file = kwargs.get("file", sys.stdout)
                flush = kwargs.get("flush", False)
                return original_print(colored, end=end, file=file, flush=flush)

            # Default behavior for non-tagged lines
            return original_print(*args, **kwargs)
        except Exception:
            # Fail-safe: never break printing
            return original_print(*args, **kwargs)

    try:
        builtins.print = color_print
    except Exception:
        # If we cannot patch, just ignore
        pass


__all__ = [
    "install_print_colorizer",
    "format_full_line_if_tagged",
]


