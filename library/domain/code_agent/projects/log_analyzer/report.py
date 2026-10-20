"""A Report: a title and sections, each a small table; and the ReportWriter base class."""

from dataclasses import dataclass


@dataclass
class Section:
    title: str
    headers: list
    rows: list

    def column_widths(self):
        """The width of each column: its longest cell or header."""
        table = [self.headers] + self.rows
        return [max(len(str(cell)) for cell in column) for column in zip(*table)]


class Report:
    def __init__(self, title):
        self.title = title
        self.sections = []

    def add_section(self, title, headers, rows):
        """Add a table; an empty one gets a "(none)" row."""
        rows = [list(row) for row in rows] or [["(none)"] + [""] * (len(headers) - 1)]
        self.sections.append(Section(title, headers, rows))


class ReportWriter:
    """Writes each section of a report; the formats differ in write_section."""

    def __init__(self, file):
        self.file = file

    def write_report(self, report):
        for section in report.sections:
            self.write_section(section)

    def write_section(self, section):
        raise NotImplementedError
