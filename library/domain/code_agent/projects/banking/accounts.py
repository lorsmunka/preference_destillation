"""An Account holds a balance. Every change to the balance is saved and recorded as a transaction."""

from config import OVERDRAFT_LIMIT
from fees import overdraft_fee
from storage import load_accounts, save_accounts
from transactions import record_transaction
from validation import check_deposit, check_withdrawal


class Account:
    def __init__(self, number, owner, balance, kind="checking"):
        self.number = number
        self.owner = owner
        self.balance = balance
        self.kind = kind   # "checking" or "savings"

    def get_balance(self):
        return self.balance

    def deposit(self, amount):
        problem = check_deposit(amount)
        if problem:
            return problem
        self.balance += amount
        record_transaction(self.number, "deposit", amount)
        self.save()
        return None

    def withdraw(self, amount):
        problem = check_withdrawal(amount)
        if problem:
            return problem
        if self.balance - amount < OVERDRAFT_LIMIT:
            return "Not enough money in the account."
        self.balance -= amount
        record_transaction(self.number, "withdrawal", amount)
        if self.balance < 0:
            fee = overdraft_fee(self.balance)
            self.balance -= fee
            record_transaction(self.number, "fee", fee, "overdraft")
        self.save()
        return None

    def save(self):
        accounts = load_accounts()
        accounts[self.number] = {"owner": self.owner, "balance": self.balance, "kind": self.kind}
        save_accounts(accounts)


def load_account(number):
    data = load_accounts()[number]
    return Account(number, data["owner"], data["balance"], data["kind"])
