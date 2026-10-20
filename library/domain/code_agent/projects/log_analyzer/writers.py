"""Writing a Report as aligned plain text or as CSV."""

import csv

from report import ReportWriter


class TextWriter(ReportWriter):
    def write_report(self, report):
        print(report.title + "\n" + "=" * len(report.title), file=self.file)
        super().write_report(report)

    def write_section(self, section):
        widths = section.column_widths()
        print(f"\n{section.title}", file=self.file)
        for row in [section.headers] + section.rows:
            cells = [str(cell).ljust(width) for cell, width in zip(row, widths)]
            print("  ".join(cells).rstrip(), file=self.file)


class CsvWriter(ReportWriter):
    """A header row per section; the first column names the section."""

    def __init__(self, file):
        super().__init__(file)
        self.table = csv.writer(file)

    def write_section(self, section):
        self.table.writerow(["section"] + section.headers)
        for row in section.rows:
            self.table.writerow([section.title] + row)


def make_writer(output_format, file):
    return CsvWriter(file) if output_format == "csv" else TextWriter(file)
