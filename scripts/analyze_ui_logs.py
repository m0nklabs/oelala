#!/usr/bin/env python3
"""
Analyze UI client logs produced by the frontend and written to
/home/flip/oelala/logs/ui_client.log

This script summarizes counts by level, top messages and time range.
"""

import os
import json
import argparse
import collections
import datetime

DEFAULT_LOG = "/home/flip/oelala/logs/ui_client.log"


def analyze(log_path=DEFAULT_LOG, top_n=10):
    if not os.path.exists(log_path):
        print("Log file not found:", log_path)
        return 1

    counts = collections.Counter()
    messages = collections.Counter()
    times = []

    with open(log_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                counts["parse_error"] += 1
                continue

            level = entry.get("level") or entry.get("severity") or "unknown"
            counts[level] += 1

            msg = entry.get("message") or entry.get("error") or json.dumps(entry)
            messages[msg] += 1

            ts = entry.get("timestamp") or entry.get("time")
            if ts:
                try:
                    times.append(datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")))
                except Exception:
                    pass

    total = sum(counts.values())
    print(f"Log analysis for: {log_path}")
    print(f"Total parsed entries: {total}")
    print("\nCounts by level:")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")

    print(f"\nTop {top_n} messages:")
    for m, c in messages.most_common(top_n):
        print(f"  {c}x - {m[:200]}")

    if times:
        print(f"\nTime range: {min(times).isoformat()} to {max(times).isoformat()}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Summarize frontend UI logs")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Path to ui_client.log file")
    parser.add_argument("--top", type=int, default=10, help="How many top messages to show")
    args = parser.parse_args()

    return analyze(args.log, args.top)


if __name__ == "__main__":
    raise SystemExit(main())
