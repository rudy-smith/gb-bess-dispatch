"""
DuckDB layer: SQL over the project's Parquet files, read in place.

Why DuckDB. The query this layer exists for is the point-in-time join: attach
to each decision the latest forecast published strictly before it. DuckDB has
that as a primitive (ASOF JOIN). SQLite does not, and would need a correlated
subquery per row; Postgres would need a server and a load step. DuckDB reads
Parquet directly, so there is no second copy of the data to fall out of step.
The data volume (tens of thousands of rows) is not the reason; pandas copes.

Why views over an in-memory database. Each connection defines views onto the
Parquet files and holds no data of its own. Rebuilding a Parquet file therefore
changes what every later query sees, with no database file that could keep a
stale copy.

Sessions run in UTC. Every timestamp in the project is a UTC instant, and a
session timezone of London would make DuckDB render them in local time, which
is where one-hour errors come from.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.prep.decisions import DEFAULT_DECISION_TIME, calendar_table, decision_table

DATA_DIR = Path("data")
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"

# View name -> Parquet file, relative to the data directory.
VIEWS: dict[str, str] = {
    "prices": "processed/prices_full.parquet",
    "wind_forecasts": "processed/wind_forecasts.parquet",
    "demand_forecasts": "processed/demand_forecasts.parquet",
    "demand_outturn": "processed/demand_outturn.parquet",
    "generation": "processed/fuelhh.parquet",
    "point_in_time": "processed/point_in_time.parquet",
}


def connect(
    data_dir: Path | str = DATA_DIR,
    required: tuple[str, ...] = (),
) -> duckdb.DuckDBPyConnection:
    """In-memory connection with one view per Parquet file that exists.

    Files that are absent are skipped, so a partial pipeline can still be
    queried, unless the view is named in `required`, in which case its absence
    raises here rather than as an unhelpful "table does not exist" later.
    """
    data_dir = Path(data_dir)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")

    missing = []
    for name, rel in VIEWS.items():
        path = data_dir / rel
        if not path.exists():
            missing.append(name)
            continue
        # DDL cannot take bound parameters, so the path is inlined as a literal
        # with any single quote doubled, which is SQL's own escape.
        literal = path.resolve().as_posix().replace("'", "''")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{literal}')")

    absent = sorted(set(required) & set(missing))
    if absent:
        con.close()
        raise FileNotFoundError(
            f"required views {absent} have no Parquet file under {data_dir}; "
            "build them first (see README, Reproducing)"
        )
    return con


def register_decisions(
    con: duckdb.DuckDBPyConnection,
    start_date: str,
    end_date: str,
    decision_local=DEFAULT_DECISION_TIME,
) -> None:
    """Register the decision-time table for a date range as the view `decisions`.

    Built in Python rather than SQL because the clock-time-to-UTC conversion
    already exists, tested, in the calendar code. Writing it a second time in
    SQL would give two definitions of the same instant that could drift apart.
    """
    con.register("decisions", decision_table(start_date, end_date, decision_local))


def register_calendar(con: duckdb.DuckDBPyConnection, start_date: str, end_date: str) -> None:
    """Register local clock keys for every half hour as the view `calendar`.

    The range must reach back far enough to cover the longest lag a query uses,
    not just the delivery days, or the earliest lags silently come back NULL.
    """
    con.register("calendar", calendar_table(start_date, end_date))


def query(
    con: duckdb.DuckDBPyConnection, sql: str, params: list | dict | None = None
) -> pd.DataFrame:
    """Run SQL and return a DataFrame. Values are passed as bound parameters,
    positional (?) or named ($name), never formatted into the string."""
    return con.execute(sql, params or []).df()


def query_file(
    con: duckdb.DuckDBPyConnection, name: str, params: list | dict | None = None
) -> pd.DataFrame:
    """Run sql/<name>.sql. Queries live in files so they can be read, diffed and
    reviewed as SQL, not as strings inside Python."""
    return query(con, (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8"), params)
