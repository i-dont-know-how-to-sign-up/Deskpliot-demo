from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def connect_sqlite(path: str | Path, *, timeout: float = 30.0) -> sqlite3.Connection:
    """创建适合桌面端多线程读写的 SQLite 连接。"""
    connection = sqlite3.connect(path, timeout=timeout)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


@contextmanager
def sqlite_transaction(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Connection.__exit__ 仅提交事务，不会关闭句柄；统一保证异常和成功路径关闭。"""
    connection = connect_sqlite(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()
