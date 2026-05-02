"""
futures_roll_builder.py
=======================
PURPOSE
-------
Clean messy Databento MNQ futures data (which has 2-3 contracts per timestamp
plus calendar spreads) into a single continuous series with exactly ONE bar
per timestamp — always the correct front-month contract.

HOW THE ROLL SCHEDULE IS USED
------------------------------
Volume_Crossover_Date is the LAST date you trade the current contract.
From the NEXT calendar day onward, you switch to the next contract.

Example from the CSV:
  MNQH21  Volume_Crossover_Date = 2021-03-11
  MNQM21  Volume_Crossover_Date = 2021-06-10

  => Trade MNQH1 from start of data  up to and INCLUDING 2021-03-11
  => Trade MNQM1 from 2021-03-12     up to and INCLUDING 2021-06-10
  => Trade MNQU1 from 2021-06-11     ... and so on

SYMBOL FORMAT MISMATCH (handled automatically)
----------------------------------------------
Roll schedule uses 2-digit year:  MNQH21, MNQM22
Market data uses  1-digit year:   MNQH1,  MNQM2

Script converts automatically:    MNQH21 -> MNQH1

INPUT FILES
-----------
1. mnq_volume_crossover.csv  (roll schedule)
2. Your Databento market data CSV

OUTPUT
------
continuous_front_month.csv  — one row per timestamp, correct contract, roll flag
"""

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROLL_SCHEDULE_PATH = r"D:\output\mnq_volume_crossover.csv"
MARKET_DATA_PATH   = r"D:\output\glbx-mdp3-20210101-20260429.ohlcv-1m.csv"
OUTPUT_PATH        = r"D:\output\continuous_front_month1.csv"
# ── STEP 1: Load the roll schedule and build date ranges ──────────────────────

def build_active_contract_map(roll_csv_path: str) -> pd.DataFrame:
    """
    Reads the roll schedule and returns a DataFrame with columns:
        market_symbol   — 1-digit year form used in market data  (e.g. MNQH1)
        active_from     — first date this contract is the front month (date)
        active_until    — last  date this contract is the front month (date)

    Logic:
        active_from  = previous contract's Volume_Crossover_Date + 1 day
                       (or 2021-01-01 for the very first contract)
        active_until = this contract's Volume_Crossover_Date
    """
    rs = pd.read_csv(roll_csv_path)
    rs["Volume_Crossover_Date"] = pd.to_datetime(rs["Volume_Crossover_Date"]).dt.date
    rs["Expiry_Date"]           = pd.to_datetime(rs["Expiry_Date"]).dt.date

    # Convert roll-schedule symbol (MNQH21) -> market-data symbol (MNQH1)
    def to_market_sym(s):
        # s = e.g. "MNQH21"  ->  prefix "MNQ" + month_code "H" + last digit "1"
        return "MNQ" + s[3] + s[-1]

    rs["market_symbol"] = rs["Contract"].apply(to_market_sym)

    rows = []
    for i, row in rs.iterrows():
        if i == 0:
            # First contract is active from the very start of the dataset
            from_date = pd.Timestamp("2021-01-01").date()
        else:
            # Active from the day AFTER the previous contract's crossover date
            prev_crossover = rs.loc[i - 1, "Volume_Crossover_Date"]
            from_date = prev_crossover + pd.Timedelta(days=1)

        until_date = row["Volume_Crossover_Date"]

        rows.append({
            "market_symbol": row["market_symbol"],
            "roll_contract": row["Contract"],
            "active_from":   from_date,
            "active_until":  until_date,
        })

    schedule = pd.DataFrame(rows)

    print("[schedule] Active contract windows:")
    print(f"  {'Symbol':<10} {'From':<14} {'Until':<14}")
    print(f"  {'-'*10} {'-'*12} {'-'*12}")
    for _, r in schedule.iterrows():
        print(f"  {r['market_symbol']:<10} {str(r['active_from']):<14} {str(r['active_until']):<14}")

    return schedule


# ── STEP 2: Load and clean market data ────────────────────────────────────────

def load_market_data(path: str) -> pd.DataFrame:
    """
    Loads Databento CSV.
    - Removes calendar spreads (symbols containing '-')
    - Parses ts_event as UTC
    - Sorts chronologically
    """
    df = pd.read_csv(path)

    # Drop spreads / synthetics
    spread_mask = df["symbol"].str.contains("-", na=False)
    n_spreads   = spread_mask.sum()
    df = df[~spread_mask].copy()
    print(f"\n[clean]  Removed {n_spreads:,} spread/synthetic rows")
    print(f"[clean]  Remaining symbols in data: {sorted(df['symbol'].unique().tolist())}")

    # Parse timestamps
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

    # Sort
    df = df.sort_values("ts_event").reset_index(drop=True)

    print(f"[clean]  {len(df):,} total rows after spread removal")
    print(f"[clean]  Date range: {df['ts_event'].min()} -> {df['ts_event'].max()}")
    return df


# ── STEP 3: Filter to front-month contract per timestamp ──────────────────────

def build_continuous_series(
    market_df: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each contract window, select rows where:
        symbol == front-month contract
        date   is within [active_from, active_until]

    This guarantees exactly one contract per timestamp.
    Then concatenates all segments in order.
    """
    # Add a date column for easy filtering (date only, no time)
    market_df["_date"] = market_df["ts_event"].dt.date

    segments = []

    for _, win in schedule.iterrows():
        sym        = win["market_symbol"]
        from_date  = win["active_from"]
        until_date = win["active_until"]

        seg = market_df[
            (market_df["symbol"] == sym) &
            (market_df["_date"]  >= from_date) &
            (market_df["_date"]  <= until_date)
        ].copy()

        if seg.empty:
            print(f"[warn]   {sym}: NO DATA in window "
                  f"[{from_date} -> {until_date}]")
            continue

        # Mark the first bar of this contract as a roll event
        seg["roll"] = False
        seg.iloc[0, seg.columns.get_loc("roll")] = True

        segments.append(seg)

        n    = len(seg)
        days = seg["_date"].nunique()
        print(f"[seg]    {sym}: {n:>6,} bars across {days:>3} trading days  "
              f"[{from_date} -> {until_date}]")

    if not segments:
        raise RuntimeError(
            "No segments produced. "
            "Check that market_symbol values match your data's symbol column."
        )

    combined = pd.concat(segments, ignore_index=True)

    # Drop the helper date column
    combined = combined.drop(columns=["_date"])

    # Final sort and dedup (safety net — should already be clean)
    combined = (
        combined
        .sort_values("ts_event")
        .drop_duplicates(subset=["ts_event"])   # one row per timestamp
        .reset_index(drop=True)
    )

    return combined


# ── STEP 4: Validate and format output ───────────────────────────────────────

OUTPUT_COLS = ["ts_event", "symbol", "open", "high", "low", "close", "volume", "roll"]

def finalise(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only needed columns (add any extras you want here)
    available = [c for c in OUTPUT_COLS if c in df.columns]
    df = df[available].copy()

    # Hard checks
    n_dupes = df.duplicated(subset=["ts_event"]).sum()
    assert n_dupes == 0, f"FAIL: {n_dupes} duplicate timestamps found!"
    assert df["ts_event"].is_monotonic_increasing, "FAIL: timestamps not sorted!"

    return df


# ── STEP 5: Print summary ────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    rolls = df[df["roll"]]
    print("\n" + "=" * 60)
    print("  OUTPUT DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total rows         : {len(df):,}")
    print(f"  Unique timestamps  : {df['ts_event'].nunique():,}")
    print(f"  Trading date range : {df['ts_event'].min().date()} -> "
          f"{df['ts_event'].max().date()}")
    print(f"  Contracts included : {df['symbol'].unique().tolist()}")
    print(f"  Number of rolls    : {len(rolls) - 1}")
    print(f"\n  Roll events:")
    print(f"  {'Timestamp':<38} {'New Contract'}")
    print(f"  {'-'*36} {'-'*14}")
    for _, row in rolls.iloc[1:].iterrows():
        print(f"  {str(row['ts_event']):<38} {row['symbol']}")
    print("=" * 60 + "\n")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main(
    roll_schedule_path : str = ROLL_SCHEDULE_PATH,
    market_data_path   : str = MARKET_DATA_PATH,
    output_path        : str = OUTPUT_PATH,
) -> pd.DataFrame:

    print("\n=== Step 1: Build active contract schedule ===")
    schedule = build_active_contract_map(roll_schedule_path)

    print("\n=== Step 2: Load and clean market data ===")
    market_df = load_market_data(market_data_path)

    print("\n=== Step 3: Build continuous front-month series ===")
    combined = build_continuous_series(market_df, schedule)

    print("\n=== Step 4: Validate output ===")
    final_df = finalise(combined)
    print("  All integrity checks passed.")

    print("\n=== Step 5: Summary ===")
    print_summary(final_df)

    final_df.to_csv(output_path, index=False)
    print(f"  Saved -> {output_path}")

    return final_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll-schedule", default=ROLL_SCHEDULE_PATH)
    parser.add_argument("--market-data",   default=MARKET_DATA_PATH)
    parser.add_argument("--output",        default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args.roll_schedule, args.market_data, args.output)