"""A Transaction is one movement of money: a deposit, a withdrawal, a fee, interest or a transfer."""

from storage import load_transactions, save_transactions
from utils import new_id, today


class Transaction:
    def __init__(self, account_number, kind, amount, note=""):
        self.id = new_id("T")
        self.account_number = account_number
        self.kind = kind   # "deposit", "withdrawal", "fee", "interest", "transfer_in" or "transfer_out"
        self.amount = amount
        self.date = today()
        self.note = note

    def to_dict(self):
        return {"id": self.id, "account": self.account_number, "kind": self.kind,
                "amount": self.amount, "date": self.date, "note": self.note}

    def is_money_in(self):
        return self.kind in ("deposit", "interest", "transfer_in")


def record_transaction(account_number, kind, amount, note=""):
    """Make a Transaction and add it to transactions.json."""
    transaction = Transaction(account_number, kind, amount, note)
    transactions = load_transactions()
    transactions.append(transaction.to_dict())
    save_transactions(transactions)
    return transaction


def transactions_of(account_number):
    """All saved transactions of one account, oldest first."""
    return [t for t in load_transactions() if t["account"] == account_number]
