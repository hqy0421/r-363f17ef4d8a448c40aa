import json
import sqlite3
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from triad import Schema
from triad.utils.cdc import (
    CDCApplyError,
    CDCContradictionError,
    CDCError,
    SQLiteTransactionLoader,
    TableSpec,
    _normalize_value,
    _sql_type,
)


def _make_loader(path, **kwargs):
    tables = {
        "t1": TableSpec("id:long,v:str", primary_key="id"),
        "t2": TableSpec("id:long,w:double", primary_key="id"),
    }
    tables.update(kwargs)
    return SQLiteTransactionLoader(path, tables)


def test_basic_iud_interleaved_and_in_order(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    # commit of txn b arrives before anything of txn a
    loader.add_commit("b", total=2, commit_seq=1)
    assert loader.applied_seq == -1
    # events interleave across transactions and tables
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    loader.add_event("b", 0, "t1", "I", {"id": 2, "v": "y"})
    assert loader.applied_seq == -1  # commit of a is still missing
    loader.add_commit("a", total=2, commit_seq=0)
    assert loader.applied_seq == -1  # a is not complete yet
    loader.add_event("b", 1, "t2", "U", {"id": 1, "w": 2.5})
    assert loader.applied_seq == -1  # b is complete but a blocks it
    loader.add_event("a", 1, "t2", "I", {"id": 1, "w": 1.5})
    # now a is complete -> a and b are applied in source order
    assert loader.applied_seq == 1
    t1 = loader.read_table("t1")
    assert t1.schema == pa.schema([("id", pa.int64()), ("v", pa.string())])
    # rowid order proves a was applied before b
    assert t1.to_pylist() == [
        {"id": 1, "v": "x"},
        {"id": 2, "v": "y"},
    ]
    # b's update on t2 happened after a's insert
    assert loader.read_table("t2").to_pylist() == [{"id": 1, "w": 2.5}]
    loader.close()


def test_commit_arrives_first_and_empty_transaction(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    loader.add_commit("a", total=2, commit_seq=0)
    loader.add_commit("b", total=0, commit_seq=1)  # empty transaction
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    assert loader.applied_seq == -1
    loader.add_event("a", 1, "t1", "I", {"id": 2, "v": "y"})
    # a is applied, and the empty transaction b also advances the sequence
    assert loader.applied_seq == 1
    assert len(loader.read_table("t1")) == 2
    loader.close()


def test_duplicate_messages_ignored(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})  # exact duplicate
    # duplicates are compared after normalization, so "1" == 1
    loader.add_event("a", 0, "t1", "I", {"id": "1", "v": "x"})
    loader.add_commit("a", total=1, commit_seq=0)
    loader.add_commit("a", total=1, commit_seq=0)  # duplicate commit
    assert loader.applied_seq == 0
    assert loader.read_table("t1").to_pylist() == [{"id": 1, "v": "x"}]
    # re-sending messages of an applied transaction does not re-apply
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    assert loader.applied_seq == 0
    assert len(loader.read_table("t1")) == 1
    loader.close()


def test_contradictory_messages_rejected(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    # same slot, different content
    with pytest.raises(CDCContradictionError):
        loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "y"})
    # same slot, different op
    with pytest.raises(CDCContradictionError):
        loader.add_event("a", 0, "t1", "D", {"id": 1})
    # same slot, different table
    with pytest.raises(CDCContradictionError):
        loader.add_event("a", 0, "t2", "I", {"id": 1, "w": 1.0})
    loader.add_commit("a", total=1, commit_seq=0)
    # same transaction, different total
    with pytest.raises(CDCContradictionError):
        loader.add_commit("a", total=2, commit_seq=0)
    # same transaction, different commit_seq
    with pytest.raises(CDCContradictionError):
        loader.add_commit("a", total=1, commit_seq=5)
    # another transaction claiming the same commit_seq
    with pytest.raises(CDCContradictionError):
        loader.add_commit("b", total=0, commit_seq=0)
    # event seq exceeding the declared total
    with pytest.raises(CDCContradictionError):
        loader.add_event("a", 1, "t1", "I", {"id": 2, "v": "z"})
    # commit declaring a total smaller than an already received event
    loader.add_event("c", 3, "t1", "I", {"id": 9, "v": "w"})
    with pytest.raises(CDCContradictionError):
        loader.add_commit("c", total=3, commit_seq=1)
    loader.close()


def test_update_and_delete(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    loader.add_event("a", 1, "t1", "I", {"id": 2, "v": "y"})
    loader.add_commit("a", total=2, commit_seq=0)
    loader.add_event("b", 0, "t1", "U", {"id": 1, "v": "xx"})
    loader.add_event("b", 1, "t1", "D", {"id": 2})
    loader.add_event("b", 2, "t1", "U", {"id": 99, "v": "none"})  # no such row
    loader.add_event("b", 3, "t1", "D", {"id": 99})  # no such row
    loader.add_commit("b", total=4, commit_seq=1)
    assert loader.applied_seq == 1
    assert loader.read_table("t1").to_pylist() == [{"id": 1, "v": "xx"}]
    loader.close()


def test_failed_transaction_rolls_back_and_blocks_later(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    loader.add_commit("a", total=1, commit_seq=0)
    assert loader.applied_seq == 0
    # txn b fails on the second event (duplicated primary key)
    loader.add_event("b", 0, "t1", "I", {"id": 2, "v": "y"})
    loader.add_event("b", 1, "t1", "I", {"id": 2, "v": "z"})
    with pytest.raises(CDCApplyError):
        loader.add_commit("b", total=2, commit_seq=1)
    # the failure does not expose a half applied transaction
    assert loader.applied_seq == 0
    assert loader.read_table("t1").to_pylist() == [{"id": 1, "v": "x"}]
    # the failed commit record is persisted
    raw = sqlite3.connect(path)
    assert raw.execute("SELECT applied_seq FROM _triad_cdc_state").fetchone() == (0,)
    assert raw.execute(
        "SELECT COUNT(*) FROM _triad_cdc_commits WHERE txn_id='b'"
    ).fetchone() == (1,)
    raw.close()
    # a later complete transaction can be received but can't jump over b;
    # every ingestion retries the failed predecessor b and raises
    with pytest.raises(CDCApplyError):
        loader.add_event("c", 0, "t1", "I", {"id": 3, "v": "w"})
    with pytest.raises(CDCApplyError):
        loader.add_commit("c", total=1, commit_seq=2)
    assert loader.applied_seq == 0
    assert loader.read_table("t1").to_pylist() == [{"id": 1, "v": "x"}]
    # explicit retry hits the same failure and keeps previous commits
    with pytest.raises(CDCApplyError):
        loader.apply_pending()
    assert loader.applied_seq == 0
    assert loader.read_table("t1").to_pylist() == [{"id": 1, "v": "x"}]
    loader.close()


def test_deferred_foreign_keys(tmp_path):
    path = str(tmp_path / "t.db")
    loader = SQLiteTransactionLoader(
        path,
        {
            "parent": TableSpec("id:long,name:str", primary_key="id"),
            "child": TableSpec(
                "id:long,pid:long",
                primary_key="id",
                foreign_keys=[("pid", "parent", "id")],
            ),
        },
    )
    # child row comes before its parent row inside the same transaction
    loader.add_event("a", 0, "child", "I", {"id": 1, "pid": 100})
    loader.add_event("a", 1, "parent", "I", {"id": 100, "name": "p"})
    loader.add_commit("a", total=2, commit_seq=0)
    assert loader.applied_seq == 0
    assert loader.read_table("child").to_pylist() == [{"id": 1, "pid": 100}]
    # a transaction violating the foreign key at the end fails and rolls back
    loader.add_event("b", 0, "parent", "I", {"id": 200, "name": "q"})
    loader.add_event("b", 1, "child", "I", {"id": 2, "pid": 999})
    with pytest.raises(CDCApplyError):
        loader.add_commit("b", total=2, commit_seq=1)
    assert loader.applied_seq == 0
    # neither the bad child nor the good parent insert of txn b is visible
    assert loader.read_table("parent").to_pylist() == [{"id": 100, "name": "p"}]
    assert loader.read_table("child").to_pylist() == [{"id": 1, "pid": 100}]
    loader.close()


def test_persistence_and_reopen(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    loader.add_commit("a", total=1, commit_seq=0)
    loader.add_event("b", 0, "t1", "I", {"id": 2, "v": "y"})  # incomplete txn
    assert loader.applied_seq == 0
    loader.close()

    # reopening continues the assembly and does not re-apply
    loader = _make_loader(path)
    assert loader.applied_seq == 0
    assert loader.read_table("t1").to_pylist() == [{"id": 1, "v": "x"}]
    loader.add_event("b", 1, "t2", "I", {"id": 5, "w": 0.5})
    loader.add_commit("b", total=2, commit_seq=1)
    assert loader.applied_seq == 1
    assert loader.read_table("t1").to_pylist() == [
        {"id": 1, "v": "x"},
        {"id": 2, "v": "y"},
    ]
    loader.close()

    # reopen again: state is kept, re-sent messages are ignored
    loader = _make_loader(path)
    assert loader.applied_seq == 1
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    loader.add_commit("a", total=1, commit_seq=0)
    assert loader.applied_seq == 1
    assert len(loader.read_table("t1")) == 2
    assert loader.read_table("t2").to_pylist() == [{"id": 5, "w": 0.5}]
    loader.close()


def test_schema_normalization_and_typed_readback(tmp_path):
    path = str(tmp_path / "t.db")
    schema = pa.schema(
        [
            pa.field("i", pa.int64()),
            pa.field("f", pa.float64()),
            pa.field("s", pa.string()),
            pa.field("b", pa.bool_()),
            pa.field("ts", pa.timestamp("us")),
            pa.field("tz", pa.timestamp("us", "UTC")),
            pa.field("d", pa.date32()),
            pa.field("bin", pa.binary()),
            pa.field("dec", pa.decimal128(10, 2)),
            pa.field("st", pa.struct([pa.field("x", pa.int32())])),
            pa.field("li", pa.list_(pa.field("item", pa.int64()))),
            pa.field("mp", pa.map_(pa.string(), pa.int64())),
        ]
    )
    loader = SQLiteTransactionLoader(path, {"t": TableSpec(schema, primary_key="i")})
    loader.add_event(
        "a",
        0,
        "t",
        "I",
        {
            "i": "42",  # normalized to int
            "f": "2.5",  # normalized to float
            "s": 100,  # normalized to str
            "b": "true",  # normalized to bool
            "ts": "2026-01-02 03:04:05",
            "tz": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            "d": "2026-01-02",
            "bin": "abc",
            "dec": "1.25",
            "st": '{"x": 1}',
            "li": "[1, 2]",
            "mp": {"k": 7},
        },
    )
    loader.add_commit("a", total=1, commit_seq=0)
    assert loader.applied_seq == 0
    df = loader.read_table("t")
    assert df.schema == schema
    assert df.to_pylist() == [
        {
            "i": 42,
            "f": 2.5,
            "s": "100",
            "b": True,
            "ts": datetime(2026, 1, 2, 3, 4, 5),
            "tz": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            "d": date(2026, 1, 2),
            "bin": b"abc",
            "dec": Decimal("1.25"),
            "st": {"x": 1},
            "li": [1, 2],
            "mp": [("k", 7)],
        }
    ]
    loader.close()


def test_read_empty_table(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    df = loader.read_table("t1")
    assert df.schema == pa.schema([("id", pa.int64()), ("v", pa.string())])
    assert df.num_rows == 0
    loader.close()


def test_validation_errors(tmp_path):
    path = str(tmp_path / "t.db")
    loader = SQLiteTransactionLoader(path, {"nopk": "id:long,v:str"})
    with pytest.raises(ValueError):
        loader.add_event("a", 0, "unknown", "I", {"id": 1})
    with pytest.raises(ValueError):
        loader.add_event("a", 0, "nopk", "X", {"id": 1})
    with pytest.raises(ValueError):
        loader.add_event("a", -1, "nopk", "I", {"id": 1})
    with pytest.raises(ValueError):
        loader.add_event("a", True, "nopk", "I", {"id": 1})
    with pytest.raises(ValueError):
        loader.add_event("", 0, "nopk", "I", {"id": 1})
    with pytest.raises(ValueError):
        loader.add_event("a", 0, "nopk", "I", [{"id": 1}])
    with pytest.raises(ValueError):
        loader.add_event("a", 0, "nopk", "I", {"id": 1, "unknown": 2})
    # update/delete require a primary key
    with pytest.raises(ValueError):
        loader.add_event("a", 0, "nopk", "U", {"id": 1})
    with pytest.raises(ValueError):
        loader.add_event("a", 0, "nopk", "D", {"id": 1})
    with pytest.raises(ValueError):
        loader.add_commit("a", -1, 0)
    with pytest.raises(ValueError):
        loader.add_commit("a", 0, -1)
    loader.close()

    loader = _make_loader(str(tmp_path / "t2.db"))
    # primary key can't be null
    with pytest.raises(ValueError):
        loader.add_event("a", 0, "t1", "I", {"v": "x"})
    with pytest.raises(ValueError):
        loader.add_event("a", 0, "t1", "D", {"v": "x"})
    # bad value that can't be normalized
    with pytest.raises(ValueError):
        loader.add_event("a", 0, "t1", "I", {"id": "not an int", "v": "x"})
    # unknown table in read
    with pytest.raises(ValueError):
        loader.read_table("unknown")
    loader.close()


def test_context_manager_and_tables_property(tmp_path):
    path = str(tmp_path / "t.db")
    with _make_loader(path) as loader:
        assert sorted(loader.tables) == ["t1", "t2"]
        loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
        loader.add_commit("a", total=1, commit_seq=0)
        assert loader.applied_seq == 0
    # the data survives after close
    raw = sqlite3.connect(path)
    assert raw.execute('SELECT id, v FROM "t1"').fetchall() == [(1, "x")]
    raw.close()


def test_reopen_with_failed_transaction(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    loader.add_commit("a", total=1, commit_seq=0)
    loader.add_event("b", 0, "t1", "I", {"id": 1, "v": "y"})  # conflicts with a
    with pytest.raises(CDCApplyError):
        loader.add_commit("b", total=1, commit_seq=1)
    loader.close()
    # reopening keeps the state: b is still pending and still failing,
    # previously committed data is preserved
    loader = _make_loader(path)
    assert loader.applied_seq == 0
    assert loader.read_table("t1").to_pylist() == [{"id": 1, "v": "x"}]
    with pytest.raises(CDCApplyError):
        loader.apply_pending()
    assert loader.applied_seq == 0
    loader.close()


def test_apply_pending_return_value(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    assert loader.apply_pending() == 0
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    loader.add_event("b", 0, "t1", "I", {"id": 2, "v": "y"})
    loader.add_commit("b", total=1, commit_seq=1)
    loader.add_commit("a", total=1, commit_seq=0)
    # both a and b were applied by the last add_commit
    assert loader.applied_seq == 1
    assert loader.apply_pending() == 0
    loader.close()


def test_update_on_all_pk_table_is_no_op(tmp_path):
    path = str(tmp_path / "t.db")
    loader = SQLiteTransactionLoader(
        path, {"t": TableSpec("id:long", primary_key="id")}
    )
    loader.add_event("a", 0, "t", "I", {"id": 1})
    loader.add_commit("a", total=1, commit_seq=0)
    loader.add_event("b", 0, "t", "U", {"id": 1})  # nothing to update
    loader.add_commit("b", total=1, commit_seq=1)
    assert loader.applied_seq == 1
    assert loader.read_table("t").to_pylist() == [{"id": 1}]
    loader.close()


def test_nullable_columns_round_trip(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    # omitted columns become null and read back as null
    loader.add_event("a", 0, "t1", "I", {"id": 1})
    loader.add_commit("a", total=1, commit_seq=0)
    assert loader.read_table("t1").to_pylist() == [{"id": 1, "v": None}]
    loader.close()


def test_table_spec():
    spec = TableSpec("a:long,b:str", primary_key=["a"])
    assert spec.primary_key == ["a"]
    assert spec.schema == Schema("a:long,b:str")
    assert spec.foreign_keys == []
    assert "TableSpec" in repr(spec)
    spec = TableSpec("a:long", foreign_keys=[("a", "t", "x")])
    assert spec.foreign_keys == [(["a"], "t", ["x"])]
    with pytest.raises(ValueError):
        TableSpec("a:long", primary_key="zz")
    with pytest.raises(ValueError):
        TableSpec("a:long,b:long", foreign_keys=[(("a", "b"), "t", "x")])
    with pytest.raises(ValueError):
        TableSpec("a:long", foreign_keys=[("zz", "t", "x")])


def test_loader_init_validation(tmp_path):
    with pytest.raises(ValueError):
        SQLiteTransactionLoader(str(tmp_path / "a.db"), {})
    with pytest.raises(ValueError):
        SQLiteTransactionLoader(str(tmp_path / "b.db"), {"_triad_cdc_x": "a:long"})


def test_corrupted_transaction_detected(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    # simulate external tampering: an event slot violating the declared total
    loader._conn.execute(
        "INSERT INTO _triad_cdc_messages (txn_id, seq, table_name, op, payload) "
        "VALUES ('a', 5, 't1', 'I', '{}')"
    )
    loader._conn.execute(
        "INSERT INTO _triad_cdc_commits (txn_id, total, commit_seq) VALUES ('a', 1, 0)"
    )
    with pytest.raises(CDCError, match="corrupted"):
        loader.apply_pending()
    loader.close()


def test_normalize_value_branches():
    assert _normalize_value(pa.null(), 1) is None
    # bool
    assert _normalize_value(pa.bool_(), True) is True
    assert _normalize_value(pa.bool_(), float("nan")) is None
    assert _normalize_value(pa.bool_(), 1) is True
    assert _normalize_value(pa.bool_(), np.int64(0)) is False
    assert _normalize_value(pa.bool_(), "0") is False
    with pytest.raises(ValueError):
        _normalize_value(pa.bool_(), "abc")
    # int
    assert _normalize_value(pa.int64(), float("nan")) is None
    assert _normalize_value(pa.int32(), 2.7) == 2
    assert _normalize_value(pa.int64(), np.float64(3.0)) == 3
    with pytest.raises(ValueError):
        _normalize_value(pa.int64(), "x")
    with pytest.raises(ValueError):
        _normalize_value(pa.int64(), [1])
    # float
    assert _normalize_value(pa.float64(), float("nan")) is None
    assert _normalize_value(pa.float64(), "1.5") == 1.5
    with pytest.raises(ValueError):
        _normalize_value(pa.float64(), [1])
    # string
    assert _normalize_value(pa.string(), 1.5) == "1.5"
    # timestamp
    assert _normalize_value(pa.timestamp("us"), pd.Timestamp("2026-01-01")) == datetime(
        2026, 1, 1
    )
    assert _normalize_value(pa.timestamp("us"), date(2026, 1, 1)) == datetime(
        2026, 1, 1
    )
    with pytest.raises(ValueError):
        _normalize_value(pa.timestamp("us"), 123)
    # date
    assert _normalize_value(pa.date32(), pd.Timestamp("2026-01-01")) == date(2026, 1, 1)
    assert _normalize_value(pa.date32(), datetime(2026, 1, 1, 5)) == date(2026, 1, 1)
    assert _normalize_value(pa.date32(), date(2026, 1, 1)) == date(2026, 1, 1)
    with pytest.raises(ValueError):
        _normalize_value(pa.date32(), 123)
    # decimal
    dec = pa.decimal128(10, 2)
    assert _normalize_value(dec, Decimal("1.5")) == Decimal("1.5")
    assert _normalize_value(dec, float("nan")) is None
    assert _normalize_value(dec, 2) == Decimal(2)
    assert _normalize_value(dec, 2.5) == Decimal("2.5")
    with pytest.raises(ValueError):
        _normalize_value(dec, [1])
    # binary
    assert _normalize_value(pa.binary(), b"ab") == b"ab"
    assert _normalize_value(pa.binary(), bytearray(b"ab")) == b"ab"
    assert _normalize_value(pa.binary(), memoryview(b"ab")) == b"ab"
    with pytest.raises(ValueError):
        _normalize_value(pa.binary(), 123)
    # struct
    st = pa.struct([pa.field("x", pa.int64())])
    assert _normalize_value(st, {"x": "1"}) == {"x": 1}
    with pytest.raises(ValueError):
        _normalize_value(st, [1])
    # list
    li = pa.list_(pa.field("item", pa.int64()))
    assert _normalize_value(li, np.array([1, 2])) == [1, 2]
    with pytest.raises(ValueError):
        _normalize_value(li, 123)
    # map
    mp = pa.map_(pa.string(), pa.int64())
    assert _normalize_value(mp, [("a", 1)]) == [("a", 1)]
    assert _normalize_value(mp, '{"a": 1}') == [("a", 1)]
    with pytest.raises(ValueError):
        _normalize_value(mp, 123)
    # unsupported types
    with pytest.raises(NotImplementedError):
        _normalize_value(pa.time32("s"), 1)
    with pytest.raises(NotImplementedError):
        _sql_type(pa.time32("s"))


def test_messages_persisted_as_json(tmp_path):
    path = str(tmp_path / "t.db")
    loader = _make_loader(path)
    loader.add_event("a", 0, "t1", "I", {"id": 1, "v": "x"})
    loader.close()
    raw = sqlite3.connect(path)
    rows = raw.execute(
        "SELECT txn_id, seq, table_name, op, payload FROM _triad_cdc_messages"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][:4] == ("a", 0, "t1", "I")
    assert json.loads(rows[0][4]) == {"id": 1, "v": "x"}
    raw.close()
