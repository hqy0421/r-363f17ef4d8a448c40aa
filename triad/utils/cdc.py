import base64
import json
import math
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw


class CDCError(Exception):
    """Base exception for CDC transaction loading errors"""

    def __init__(self, message: Any):
        super().__init__(message)


class CDCContradictionError(CDCError):
    """Two messages occupy the same slot with different content"""

    def __init__(self, message: Any):
        super().__init__(message)


class CDCApplyError(CDCError):
    """A complete transaction failed to be applied to the target"""

    def __init__(self, message: Any):
        super().__init__(message)


class TableSpec:
    """The specification of a CDC target table

    :param schema: a schema like object, see
        :class:`~triad.collections.schema.Schema`
    :param primary_key: primary key column name(s). It is required if the
        stream contains ``U`` (update) or ``D`` (delete) events on this table
    :param foreign_keys: an iterable of ``(columns, referred_table,
        referred_columns)`` tuples. Foreign keys are enforced by SQLite at the
        end of each source transaction (deferred), so inside one transaction
        child rows may appear before their parent rows

    .. admonition:: Examples

        .. code-block:: python

            TableSpec("id:long,name:str", primary_key="id")
            TableSpec(
                "id:long,order_id:long",
                primary_key="id",
                foreign_keys=[("order_id", "orders", "id")],
            )
    """

    def __init__(
        self,
        schema: Any,
        primary_key: Optional[Union[str, Iterable[str]]] = None,
        foreign_keys: Optional[
            Iterable[Tuple[Union[str, Iterable[str]], str, Union[str, Iterable[str]]]]
        ] = None,
    ):
        self._schema = Schema(schema).assert_not_empty()
        if primary_key is None:
            self._primary_key: List[str] = []
        elif isinstance(primary_key, str):
            self._primary_key = [primary_key]
        else:
            self._primary_key = list(primary_key)
        for k in self._primary_key:
            assert_or_throw(
                k in self._schema,
                ValueError(f"primary key column {k} is not in {self._schema}"),
            )
        self._foreign_keys: List[Tuple[List[str], str, List[str]]] = []
        for fk in foreign_keys or []:
            cols, ref_table, ref_cols = fk
            cols = [cols] if isinstance(cols, str) else list(cols)
            ref_cols = [ref_cols] if isinstance(ref_cols, str) else list(ref_cols)
            assert_or_throw(
                len(cols) == len(ref_cols),
                ValueError(f"foreign key {fk} has mismatched columns"),
            )
            for c in cols:
                assert_or_throw(
                    c in self._schema,
                    ValueError(f"foreign key column {c} is not in {self._schema}"),
                )
            self._foreign_keys.append((cols, ref_table, ref_cols))

    @property
    def schema(self) -> Schema:
        """The declared schema of the table"""
        return self._schema

    @property
    def primary_key(self) -> List[str]:
        """The primary key column names"""
        return list(self._primary_key)

    @property
    def foreign_keys(self) -> List[Tuple[List[str], str, List[str]]]:
        """Foreign keys as ``(columns, referred_table, referred_columns)``"""
        return list(self._foreign_keys)

    def __repr__(self) -> str:
        return (
            f"TableSpec({self._schema}, primary_key={self._primary_key}, "
            f"foreign_keys={self._foreign_keys})"
        )


class SQLiteTransactionLoader:
    """SQLite backed transactional loader for interleaved multi-table CDC
    streams.

    Data events of different source transactions may interleave across
    batches, and commit records may arrive before all data events of their
    transaction. This loader persists everything it receives, assembles
    transactions, and once a transaction is complete (its commit record and
    all of its declared events have been collected) applies it to the target
    SQLite database. Transactions are applied strictly in source commit
    order: a later transaction can never be applied when an earlier one is
    still missing or has failed.

    * Data events are identified by ``(txn_id, seq)`` where ``seq`` is a
      0-based consecutive event sequence number inside the transaction
    * Commit records declare the total number of events of the transaction
      (``total``) and a 0-based consecutive source commit sequence number
      (``commit_seq``)
    * Empty transactions (``total == 0``) also advance the applied sequence
    * Duplicated messages are ignored; a message contradicting an existing
      slot raises :class:`~triad.utils.cdc.CDCContradictionError`
    * Received events and commits are persisted in the same SQLite file, so
      reopening the loader continues the assembly and never re-applies
      transactions that were already applied
    * Target table changes and the applied commit sequence number are
      committed in one SQLite transaction, so a failure never exposes a
      half applied transaction and previously committed transactions are
      always preserved
    * Rows are normalized to the declared table schemas on ingestion
    * Foreign keys are checked at the end of each transaction (deferred), so
      child rows may be applied before their parent rows inside the same
      transaction

    :param path: the path of the SQLite database file (``":memory:"`` also
        works)
    :param tables: a dictionary of table name to
        :class:`~triad.utils.cdc.TableSpec` or a schema like object (the
        latter means a table without primary key and foreign keys)

    .. admonition:: Examples

        .. code-block:: python

            with SQLiteTransactionLoader(
                "/tmp/cdc.db",
                {"orders": TableSpec("id:long,amount:double", primary_key="id")},
            ) as loader:
                loader.add_event("t1", 0, "orders", "I", {"id": 1, "amount": 2.5})
                loader.add_commit("t1", total=1, commit_seq=0)
                assert loader.applied_seq == 0
                df = loader.read_table("orders")  # pyarrow.Table
    """

    _MESSAGE_TABLE = "_triad_cdc_messages"
    _COMMIT_TABLE = "_triad_cdc_commits"
    _STATE_TABLE = "_triad_cdc_state"

    def __init__(self, path: str, tables: Dict[str, Any]):
        assert_or_throw(len(tables) > 0, ValueError("tables can't be empty"))
        self._tables: Dict[str, TableSpec] = {}
        for name, spec in tables.items():
            assert_or_throw(
                not name.startswith("_triad_cdc_"),
                ValueError(f"{name} is a reserved table name"),
            )
            self._tables[name] = (
                spec if isinstance(spec, TableSpec) else TableSpec(spec)
            )
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_metadata()
        for name, spec in self._tables.items():
            self._init_target(name, spec)

    @property
    def tables(self) -> List[str]:
        """The declared target table names"""
        return list(self._tables)

    @property
    def applied_seq(self) -> int:
        """The source commit sequence number of the last applied transaction.
        It is -1 when nothing has been applied
        """
        row = self._conn.execute(
            f"SELECT applied_seq FROM {self._STATE_TABLE} WHERE id=0"
        ).fetchone()
        return int(row[0])

    def add_event(
        self,
        txn_id: str,
        seq: int,
        table: str,
        op: str,
        values: Dict[str, Any],
    ) -> None:
        """Add one data event of a transaction. If the addition makes one or
        more transactions applicable, they are applied in source commit order.

        :param txn_id: the source transaction id
        :param seq: the 0-based consecutive event sequence number inside the
            transaction
        :param table: the target table name (must be declared)
        :param op: ``"I"`` (insert), ``"U"`` (update) or ``"D"`` (delete)
        :param values: the row values as a dictionary. For ``I`` and ``U`` it
            is the full row image (missing columns become ``None``); for
            ``D`` only the primary key columns are used. Values are
            normalized to the declared table schema.

        :raises ValueError: if the input is invalid
        :raises CDCContradictionError: if the slot ``(txn_id, seq)`` is
            already occupied by a different event, or the event exceeds the
            total declared by an existing commit record
        :raises CDCApplyError: if applying a complete transaction fails
        """
        spec = self._get_spec(table)
        _validate_txn_id(txn_id)
        _validate_non_negative_int("seq", seq)
        op = _validate_op(op)
        row = self._normalize_row(spec, table, op, values)
        payload = json.dumps(
            {f.name: _to_jsonable(f.type, row[f.name]) for f in spec.schema.fields},
            sort_keys=True,
        )
        existing = self._conn.execute(
            f"SELECT table_name, op, payload FROM {self._MESSAGE_TABLE} "
            "WHERE txn_id=? AND seq=?",
            (txn_id, seq),
        ).fetchone()
        if existing is not None:
            if (existing[0], existing[1], existing[2]) == (table, op, payload):
                return  # duplicated message, ignore
            raise CDCContradictionError(
                f"event slot ({txn_id!r}, {seq}) is already occupied by "
                f"({existing[0]!r}, {existing[1]!r}), can't accept ({table!r}, {op!r})"
            )
        commit = self._conn.execute(
            f"SELECT total FROM {self._COMMIT_TABLE} WHERE txn_id=?", (txn_id,)
        ).fetchone()
        if commit is not None and seq >= commit[0]:
            raise CDCContradictionError(
                f"event seq {seq} exceeds the total {commit[0]} declared by "
                f"the commit record of transaction {txn_id!r}"
            )
        self._conn.execute(
            f"INSERT INTO {self._MESSAGE_TABLE} "
            "(txn_id, seq, table_name, op, payload) VALUES (?,?,?,?,?)",
            (txn_id, seq, table, op, payload),
        )
        self.apply_pending()

    def add_commit(self, txn_id: str, total: int, commit_seq: int) -> None:
        """Add the commit record of a transaction. If the addition makes one
        or more transactions applicable, they are applied in source commit
        order.

        :param txn_id: the source transaction id
        :param total: the total number of data events of the transaction
            (0 for an empty transaction)
        :param commit_seq: the 0-based consecutive source commit sequence
            number

        :raises ValueError: if the input is invalid
        :raises CDCContradictionError: if a different commit record exists
            for the same transaction, another transaction owns the same
            ``commit_seq``, or already received events exceed ``total``
        :raises CDCApplyError: if applying a complete transaction fails
        """
        _validate_txn_id(txn_id)
        _validate_non_negative_int("total", total)
        _validate_non_negative_int("commit_seq", commit_seq)
        existing = self._conn.execute(
            f"SELECT total, commit_seq FROM {self._COMMIT_TABLE} WHERE txn_id=?",
            (txn_id,),
        ).fetchone()
        if existing is not None:
            if (existing[0], existing[1]) == (total, commit_seq):
                return  # duplicated commit, ignore
            raise CDCContradictionError(
                f"transaction {txn_id!r} already has commit record "
                f"(total={existing[0]}, commit_seq={existing[1]}), can't accept "
                f"(total={total}, commit_seq={commit_seq})"
            )
        owner = self._conn.execute(
            f"SELECT txn_id FROM {self._COMMIT_TABLE} WHERE commit_seq=?",
            (commit_seq,),
        ).fetchone()
        if owner is not None:
            raise CDCContradictionError(
                f"commit_seq {commit_seq} is already owned by transaction "
                f"{owner[0]!r}, can't be claimed by {txn_id!r}"
            )
        max_seq = self._conn.execute(
            f"SELECT MAX(seq) FROM {self._MESSAGE_TABLE} WHERE txn_id=?", (txn_id,)
        ).fetchone()[0]
        if max_seq is not None and max_seq >= total:
            raise CDCContradictionError(
                f"transaction {txn_id!r} already has event seq {max_seq}, "
                f"which exceeds the declared total {total}"
            )
        self._conn.execute(
            f"INSERT INTO {self._COMMIT_TABLE} (txn_id, total, commit_seq) "
            "VALUES (?,?,?)",
            (txn_id, total, commit_seq),
        )
        self.apply_pending()

    def apply_pending(self) -> int:
        """Apply all transactions that are complete and next in source commit
        order. A transaction is applicable only when its commit record and
        all of its declared events have been collected, and every earlier
        transaction (by ``commit_seq``) has been applied. A missing or failed
        predecessor blocks all later transactions.

        :raises CDCApplyError: if applying a complete transaction fails. The
            failed transaction is rolled back entirely, previously committed
            transactions are preserved, and the same transaction will be
            retried on the next call
        :return: the number of transactions applied in this call
        """
        applied = 0
        while True:
            nxt = self.applied_seq + 1
            commit = self._conn.execute(
                f"SELECT txn_id, total FROM {self._COMMIT_TABLE} WHERE commit_seq=?",
                (nxt,),
            ).fetchone()
            if commit is None:
                break
            txn_id, total = commit[0], commit[1]
            events = self._conn.execute(
                f"SELECT seq, table_name, op, payload FROM {self._MESSAGE_TABLE} "
                "WHERE txn_id=? ORDER BY seq",
                (txn_id,),
            ).fetchall()
            if len(events) < total:
                break  # the transaction is not complete yet
            self._apply_transaction(txn_id, nxt, total, events)
            applied += 1
        return applied

    def read_table(self, table: str) -> pa.Table:
        """Read the current content of a target table as a typed pyarrow
        table, in the declared schema

        :param table: the target table name (must be declared)
        :raises ValueError: if the table is not declared
        :return: the pyarrow table in the declared schema
        """
        spec = self._get_spec(table)
        names = spec.schema.names
        types = spec.schema.types
        cols = ", ".join(_quote(n) for n in names)
        rows = []
        for rec in self._conn.execute(
            f"SELECT {cols} FROM {_quote(table)} ORDER BY rowid"
        ):
            rows.append(
                {n: _from_storage(tp, v) for n, tp, v in zip(names, types, rec)}
            )
        if len(rows) == 0:
            return spec.schema.create_empty_arrow_table()
        return pa.Table.from_pylist(rows, schema=spec.schema.pa_schema)

    def close(self) -> None:
        """Close the underlying SQLite connection"""
        self._conn.close()

    def __enter__(self) -> "SQLiteTransactionLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _get_spec(self, table: str) -> TableSpec:
        assert_or_throw(
            table in self._tables, ValueError(f"{table} is not a declared table")
        )
        return self._tables[table]

    def _init_metadata(self) -> None:
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._MESSAGE_TABLE} ("
            "txn_id TEXT NOT NULL, "
            "seq INTEGER NOT NULL, "
            "table_name TEXT NOT NULL, "
            "op TEXT NOT NULL, "
            "payload TEXT NOT NULL, "
            "PRIMARY KEY (txn_id, seq))"
        )
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._COMMIT_TABLE} ("
            "txn_id TEXT NOT NULL PRIMARY KEY, "
            "total INTEGER NOT NULL, "
            "commit_seq INTEGER NOT NULL)"
        )
        self._conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {self._COMMIT_TABLE}_seq "
            f"ON {self._COMMIT_TABLE} (commit_seq)"
        )
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._STATE_TABLE} ("
            "id INTEGER NOT NULL PRIMARY KEY CHECK (id=0), "
            "applied_seq INTEGER NOT NULL)"
        )
        self._conn.execute(
            f"INSERT OR IGNORE INTO {self._STATE_TABLE} (id, applied_seq) "
            "VALUES (0, -1)"
        )

    def _init_target(self, name: str, spec: TableSpec) -> None:
        pk = set(spec.primary_key)
        defs = []
        for f in spec.schema.fields:
            not_null = " NOT NULL" if not f.nullable or f.name in pk else ""
            defs.append(f"{_quote(f.name)} {_sql_type(f.type)}{not_null}")
        if len(spec.primary_key) > 0:
            defs.append(
                "PRIMARY KEY (" + ", ".join(_quote(k) for k in spec.primary_key) + ")"
            )
        for cols, ref_table, ref_cols in spec.foreign_keys:
            defs.append(
                "FOREIGN KEY ("
                + ", ".join(_quote(c) for c in cols)
                + ") REFERENCES "
                + _quote(ref_table)
                + " ("
                + ", ".join(_quote(c) for c in ref_cols)
                + ")"
            )
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_quote(name)} ({', '.join(defs)})"
        )

    def _normalize_row(
        self, spec: TableSpec, table: str, op: str, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert_or_throw(
            isinstance(values, Dict),
            ValueError(f"values of ({table}, {op}) must be a dictionary"),
        )
        unknown = sorted(set(values) - set(spec.schema.names))
        assert_or_throw(
            len(unknown) == 0,
            ValueError(f"columns {unknown} are not in the schema of table {table}"),
        )
        if op in ("U", "D"):
            assert_or_throw(
                len(spec.primary_key) > 0,
                ValueError(f"table {table} has no primary key, {op} is not allowed"),
            )
        row = {
            f.name: _normalize_value(f.type, values.get(f.name))
            for f in spec.schema.fields
        }
        if len(spec.primary_key) > 0:
            missing = [k for k in spec.primary_key if row[k] is None]
            assert_or_throw(
                len(missing) == 0,
                ValueError(f"primary key columns {missing} of table {table} are null"),
            )
        return row

    def _apply_transaction(
        self, txn_id: str, commit_seq: int, total: int, events: List[Tuple]
    ) -> None:
        assert_or_throw(
            len(events) == total and all(e[0] == i for i, e in enumerate(events)),
            CDCError(f"transaction {txn_id!r} is corrupted"),
        )
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        cur.execute("PRAGMA defer_foreign_keys=ON")
        try:
            for _, table, op, payload in events:
                spec = self._tables[table]
                data = json.loads(payload)
                values = {
                    f.name: _from_jsonable(f.type, data[f.name])
                    for f in spec.schema.fields
                }
                self._apply_op(cur, table, spec, op, values)
            cur.execute(
                f"UPDATE {self._STATE_TABLE} SET applied_seq=? WHERE id=0",
                (commit_seq,),
            )
            cur.execute("COMMIT")
        except Exception as e:
            try:
                cur.execute("ROLLBACK")
            except Exception:  # pragma: no cover
                pass
            if isinstance(e, CDCError):  # pragma: no cover
                raise
            raise CDCApplyError(
                f"failed to apply transaction {txn_id!r} (commit_seq={commit_seq}): {e}"
            ) from e

    def _apply_op(
        self,
        cur: sqlite3.Cursor,
        table: str,
        spec: TableSpec,
        op: str,
        values: Dict[str, Any],
    ) -> None:
        names = spec.schema.names
        types = spec.schema.types
        if op == "I":
            cols = ", ".join(_quote(n) for n in names)
            ph = ", ".join("?" * len(names))
            cur.execute(
                f"INSERT INTO {_quote(table)} ({cols}) VALUES ({ph})",
                [_to_storage(tp, values[n]) for n, tp in zip(names, types)],
            )
            return
        pk = spec.primary_key
        where = " AND ".join(f"{_quote(k)}=?" for k in pk)
        pk_values = [_to_storage(spec.schema[k].type, values[k]) for k in pk]
        if op == "U":
            non_pk = [n for n in names if n not in pk]
            if len(non_pk) == 0:
                return
            setter = ", ".join(f"{_quote(n)}=?" for n in non_pk)
            params = [
                _to_storage(spec.schema[n].type, values[n]) for n in non_pk
            ] + pk_values
            cur.execute(f"UPDATE {_quote(table)} SET {setter} WHERE {where}", params)
        else:  # "D"
            cur.execute(f"DELETE FROM {_quote(table)} WHERE {where}", pk_values)


def _validate_txn_id(txn_id: Any) -> None:
    assert_or_throw(
        isinstance(txn_id, str) and len(txn_id) > 0,
        ValueError(f"txn_id must be a non-empty string: {txn_id!r}"),
    )


def _validate_non_negative_int(name: str, value: Any) -> None:
    assert_or_throw(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        ValueError(f"{name} must be a non-negative integer: {value!r}"),
    )


def _validate_op(op: Any) -> str:
    assert_or_throw(
        isinstance(op, str) and op.upper() in ("I", "U", "D"),
        ValueError(f"op must be one of 'I', 'U', 'D': {op!r}"),
    )
    return op.upper()


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _is_nan(value: Any) -> bool:
    return isinstance(value, (float, np.floating)) and math.isnan(value)


def _normalize_value(tp: pa.DataType, value: Any) -> Any:  # noqa: C901
    """Normalize a python value to the declared pyarrow type. It follows the
    semantics of :func:`~triad.utils.pyarrow.apply_schema` with additional
    decimal support
    """
    if value is None or value is pd.NaT:
        return None
    if pa.types.is_null(tp):
        return None
    if pa.types.is_boolean(tp):
        if isinstance(value, bool):
            return value
        if _is_nan(value):
            return None
        if isinstance(value, (int, np.integer, float, np.floating)):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "t", "1", "yes", "y"):
                return True
            if s in ("false", "f", "0", "no", "n"):
                return False
        raise ValueError(f"{value!r} can't be converted to bool")
    if pa.types.is_integer(tp):
        if _is_nan(value):
            return None
        if isinstance(value, (bool, int, np.integer, float, np.floating)):
            return int(value)
        if isinstance(value, str):
            return int(value.strip())
        raise ValueError(f"{value!r} can't be converted to int")
    if pa.types.is_floating(tp):
        if _is_nan(value):
            return None
        if isinstance(value, (bool, int, np.integer, float, np.floating)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        raise ValueError(f"{value!r} can't be converted to float")
    if pa.types.is_string(tp) or pa.types.is_large_string(tp):
        return value if isinstance(value, str) else str(value)
    if pa.types.is_timestamp(tp):
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day)
        if isinstance(value, str):
            return datetime.fromisoformat(value.strip())
        raise ValueError(f"{value!r} can't be converted to datetime")
    if pa.types.is_date(tp):
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().date()
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value.strip())
        raise ValueError(f"{value!r} can't be converted to date")
    if pa.types.is_decimal(tp):
        if isinstance(value, Decimal):
            return value
        if _is_nan(value):
            return None
        if isinstance(value, (bool, int, np.integer)):
            return Decimal(int(value))
        if isinstance(value, (float, np.floating)):
            return Decimal(str(value))
        if isinstance(value, str):
            return Decimal(value.strip())
        raise ValueError(f"{value!r} can't be converted to decimal")
    if pa.types.is_binary(tp) or pa.types.is_large_binary(tp):
        if isinstance(value, bytes):
            return value
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, str):
            return value.encode("utf-8")
        raise ValueError(f"{value!r} can't be converted to bytes")
    if pa.types.is_struct(tp):
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Dict):
            raise ValueError(f"{value!r} can't be converted to struct")
        return {f.name: _normalize_value(f.type, value.get(f.name)) for f in tp}
    if pa.types.is_list(tp) or pa.types.is_large_list(tp):
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{value!r} can't be converted to list")
        return [_normalize_value(tp.value_type, x) for x in value]
    if pa.types.is_map(tp):
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, Dict):
            items = list(value.items())
        elif isinstance(value, (list, tuple)):
            items = [tuple(x) for x in value]
        else:
            raise ValueError(f"{value!r} can't be converted to map")
        return [
            (_normalize_value(tp.key_type, k), _normalize_value(tp.item_type, v))
            for k, v in items
        ]
    raise NotImplementedError(f"{tp} is not supported")


def _to_jsonable(tp: pa.DataType, value: Any) -> Any:
    """Convert a normalized value to a JSON serializable value"""
    if value is None:
        return None
    if pa.types.is_timestamp(tp) or pa.types.is_date(tp):
        return value.isoformat()
    if pa.types.is_decimal(tp):
        return str(value)
    if pa.types.is_binary(tp) or pa.types.is_large_binary(tp):
        return base64.b64encode(value).decode("ascii")
    if pa.types.is_struct(tp):
        return {f.name: _to_jsonable(f.type, value.get(f.name)) for f in tp}
    if pa.types.is_list(tp) or pa.types.is_large_list(tp):
        return [_to_jsonable(tp.value_type, x) for x in value]
    if pa.types.is_map(tp):
        return [
            [_to_jsonable(tp.key_type, k), _to_jsonable(tp.item_type, v)]
            for k, v in value
        ]
    return value


def _from_jsonable(tp: pa.DataType, value: Any) -> Any:
    """Inverse of :func:`_to_jsonable`"""
    if value is None:
        return None
    if pa.types.is_timestamp(tp):
        return datetime.fromisoformat(value)
    if pa.types.is_date(tp):
        return date.fromisoformat(value)
    if pa.types.is_decimal(tp):
        return Decimal(value)
    if pa.types.is_binary(tp) or pa.types.is_large_binary(tp):
        return base64.b64decode(value)
    if pa.types.is_struct(tp):
        return {f.name: _from_jsonable(f.type, value.get(f.name)) for f in tp}
    if pa.types.is_list(tp) or pa.types.is_large_list(tp):
        return [_from_jsonable(tp.value_type, x) for x in value]
    if pa.types.is_map(tp):
        return [
            (_from_jsonable(tp.key_type, k), _from_jsonable(tp.item_type, v))
            for k, v in value
        ]
    return value


def _to_storage(tp: pa.DataType, value: Any) -> Any:
    """Convert a normalized value to a SQLite bind value"""
    if value is None:
        return None
    if pa.types.is_boolean(tp):
        return 1 if value else 0
    if pa.types.is_timestamp(tp) or pa.types.is_date(tp):
        return value.isoformat()
    if pa.types.is_decimal(tp):
        return str(value)
    if (
        pa.types.is_struct(tp)
        or pa.types.is_list(tp)
        or pa.types.is_large_list(tp)
        or pa.types.is_map(tp)
    ):
        return json.dumps(_to_jsonable(tp, value), sort_keys=True)
    return value


def _from_storage(tp: pa.DataType, value: Any) -> Any:
    """Convert a SQLite value back to a normalized python value"""
    if value is None:
        return None
    if pa.types.is_boolean(tp):
        return bool(value)
    if pa.types.is_timestamp(tp):
        return datetime.fromisoformat(value) if isinstance(value, str) else value
    if pa.types.is_date(tp):
        return date.fromisoformat(value) if isinstance(value, str) else value
    if pa.types.is_decimal(tp):
        return Decimal(value)
    if pa.types.is_binary(tp) or pa.types.is_large_binary(tp):
        return bytes(value)
    if (
        pa.types.is_struct(tp)
        or pa.types.is_list(tp)
        or pa.types.is_large_list(tp)
        or pa.types.is_map(tp)
    ):
        return (
            _from_jsonable(tp, json.loads(value)) if isinstance(value, str) else value
        )
    return value


def _sql_type(tp: pa.DataType) -> str:
    if pa.types.is_boolean(tp) or pa.types.is_integer(tp):
        return "INTEGER"
    if pa.types.is_floating(tp):
        return "REAL"
    if pa.types.is_binary(tp) or pa.types.is_large_binary(tp):
        return "BLOB"
    if (
        pa.types.is_null(tp)
        or pa.types.is_string(tp)
        or pa.types.is_large_string(tp)
        or pa.types.is_date(tp)
        or pa.types.is_timestamp(tp)
        or pa.types.is_decimal(tp)
        or pa.types.is_struct(tp)
        or pa.types.is_list(tp)
        or pa.types.is_large_list(tp)
        or pa.types.is_map(tp)
    ):
        return "TEXT"
    raise NotImplementedError(f"{tp} is not supported")
