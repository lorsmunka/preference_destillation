"""A tiny spreadsheet in the terminal: type "A1 12" or "B1 =A1*2"."""

from cells import FormulaError
from config import CSV_FILE
from csv_files import export_csv, import_csv
from display import print_grid
from history import History
from references import REFERENCE_PATTERN
from sheet import Sheet

HELP = "A1 12 | A1 =SUM(B1:B3)*2 | A1 (clears it) | undo | import [file] | export [file] | quit"


def run_command(line, sheet, history):
    """Do one command. False when the user quits."""
    word, _, rest = line.strip().partition(" ")
    command = word.lower()
    if command == "quit":
        return False
    if command == "undo":
        print("Undone: " + (", ".join(history.undo(sheet)) or "nothing"))
    elif command == "import":
        history.record(import_csv(sheet, rest.strip() or CSV_FILE))
    elif command == "export":
        print(f"{export_csv(sheet, rest.strip() or CSV_FILE)} rows written.")
    elif REFERENCE_PATTERN.fullmatch(word.upper()):
        history.record([(word.upper(), sheet.set_cell(word, rest))])
    else:
        print(HELP)
    return True


def main():
    sheet, history = Sheet(), History()
    print(HELP)
    while True:
        print_grid(sheet)
        try:
            if not run_command(input("> "), sheet, history):
                break
        except FormulaError as error:
            print(error)
        except OSError as error:
            print(error)


if __name__ == "__main__":
    main()
