# auth.py — User Authentication & Management
# Place this file in the same folder as app.py

import json
import hashlib
import os
from datetime import datetime

USERS_FILE = "users.json"


def _hash(password: str) -> str:
    return hashlib.sha256(password.strip().encode()).hexdigest()


DEFAULT_USERS = {
    "admin": {
        "password": _hash("admin123"),
        "role": "admin",
        "name": "Admin",
        "created_at": datetime.now().isoformat(),
    },
    "user1": {
        "password": _hash("user123"),
        "role": "user",
        "name": "Demo User",
        "created_at": datetime.now().isoformat(),
    },
}


def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        save_users(DEFAULT_USERS)
        return DEFAULT_USERS
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def verify_login(username: str, password: str):
    users = load_users()
    u = users.get(username.strip())
    if u and u["password"] == _hash(password):
        return u
    return None


def add_user(username: str, password: str, name: str, role: str) -> bool:
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "password": _hash(password),
        "role": role,
        "name": name,
        "created_at": datetime.now().isoformat(),
    }
    save_users(users)
    return True


def delete_user(username: str) -> bool:
    users = load_users()
    if username not in users or username == "admin":
        return False
    del users[username]
    save_users(users)
    return True


def change_password(username: str, new_password: str):
    users = load_users()
    if username in users:
        users[username]["password"] = _hash(new_password)
        save_users(users)


def get_all_users() -> dict:
    return load_users()
