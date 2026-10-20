"""The command line: the log, the filters, where the report goes."""

import argparse
import re

from config import DEFAULT_LOG, THRESHOLDS_FILE, TOP_COUNT

STATUS_PATTERN = r"[1-5](\d\d|xx)"


def build_parser():
    parser = argparse.ArgumentParser(description="Summarise an access log.")
    parser.add_argument("log_file", nargs="?", default=DEFAULT_LOG)
    parser.add_argument("--since", help='"2026-10-21" or "2026-10-21 13:00"')
    parser.add_argument("--until", help="not included")
    parser.add_argument("--status", help="a code (404) or class (5xx)")
    parser.add_argument("--path", dest="path_prefix", default="")
    parser.add_argument("--top", type=int, default=TOP_COUNT)
    parser.add_argument("--format", dest="output_format", choices=("text", "csv"))
    parser.add_argument("--output", help="a file for the report")
    parser.add_argument("--limits", dest="limits_file", default=THRESHOLDS_FILE)
    return parser


def argument_problem(args):
    """Why the arguments cannot work, or None."""
    if args.top < 1:
        return "--top must be 1 or more"
    if args.status and not re.fullmatch(STATUS_PATTERN, args.status):
        return "--status takes a code like 404 or a class like 5xx"
    return None
