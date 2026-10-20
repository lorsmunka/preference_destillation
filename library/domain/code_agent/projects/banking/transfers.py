"""Moving money from one account to another."""

from accounts import load_account
from transactions import record_transaction
from validation import account_exists, check_withdrawal


def transfer(from_number, to_number, amount):
    """Move `amount` between two accounts. None when it worked, otherwise the reason it did not."""
    if from_number == to_number:
        return "You cannot send money to the same account."
    if not account_exists(to_number):
        return f"There is no account {to_number}."
    problem = check_withdrawal(amount)
    if problem:
        return problem
    sender = load_account(from_number)
    receiver = load_account(to_number)
    if sender.balance < amount:
        return "Not enough money for this transfer."
    sender.balance -= amount
    receiver.balance += amount
    record_transaction(from_number, "transfer_out", amount, f"to {to_number}")
    record_transaction(to_number, "transfer_in", amount, f"from {from_number}")
    sender.save()
    receiver.save()
    return None
