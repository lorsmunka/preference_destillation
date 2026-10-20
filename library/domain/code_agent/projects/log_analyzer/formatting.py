"""Numbers as short text for the report."""

SIZE_UNITS = ("B", "KB", "MB", "GB")
BAR_WIDTH = 30


def format_size(size):
    """5120 -> "5.0 KB"."""
    for unit in SIZE_UNITS:
        if size < 1024 or unit == SIZE_UNITS[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024


def format_seconds(seconds):
    """0.042 -> "42 ms", 1.5 -> "1.50 s", None -> "-"."""
    if seconds is None:
        return "-"
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def bar(value, largest):
    """A bar of # signs, BAR_WIDTH long for the largest value."""
    return "#" * round(BAR_WIDTH * value / largest) if largest else ""
