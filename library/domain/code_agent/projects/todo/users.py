"""Each person has their own list: a username and a password."""

from storage import load_users, save_users


def create_user(username, password):
    """False when the name is taken or the password is shorter than 4 letters."""
    users = load_users()
    if username in users or len(password) < 4:
        return False
    users[username] = {"password": password}
    save_users(users)
    return True


def password_matches(username, password):
    user = load_users().get(username)
    return user is not None and user["password"] == password


def login_screen():
    """Log in, or make a new user when the name is new. The username, or None."""
    username = input("Name: ").strip().lower()
    password = input("Password: ")
    if username not in load_users():
        return username if create_user(username, password) else None
    return username if password_matches(username, password) else None
