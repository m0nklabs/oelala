"""Tests for storage_utils (shared Range parser + helpers)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

import pytest
from storage_utils import parse_range_header, format_last_modified


class TestParseRangeHeader:
    """Tests for parse_range_header()."""

    def test_explicit_range(self):
        assert parse_range_header("bytes=0-499", 1000) == (0, 499)

    def test_open_ended_range(self):
        assert parse_range_header("bytes=500-", 1000) == (500, 999)

    def test_suffix_range(self):
        assert parse_range_header("bytes=-500", 1000) == (500, 999)

    def test_suffix_range_larger_than_file(self):
        assert parse_range_header("bytes=-2000", 1000) == (0, 999)

    def test_clamp_end_to_total_size(self):
        assert parse_range_header("bytes=0-9999", 1000) == (0, 999)

    def test_single_byte(self):
        assert parse_range_header("bytes=0-0", 1000) == (0, 0)

    def test_last_byte(self):
        assert parse_range_header("bytes=999-999", 1000) == (999, 999)

    def test_suffix_one_byte(self):
        assert parse_range_header("bytes=-1", 1000) == (999, 999)

    def test_none_for_missing_header(self):
        assert parse_range_header("", 1000) is None
        assert parse_range_header(None, 1000) is None

    def test_none_for_non_bytes_header(self):
        assert parse_range_header("items=0-10", 1000) is None

    def test_raises_on_unsatisfiable_start_beyond_end(self):
        with pytest.raises(ValueError):
            parse_range_header("bytes=500-100", 1000)

    def test_raises_on_start_beyond_total(self):
        with pytest.raises(ValueError):
            parse_range_header("bytes=1000-", 1000)

    def test_raises_on_zero_suffix(self):
        with pytest.raises(ValueError):
            parse_range_header("bytes=-0", 1000)

    def test_raises_on_negative_suffix(self):
        with pytest.raises(ValueError):
            parse_range_header("bytes=--5", 1000)

    def test_raises_on_multi_range(self):
        with pytest.raises(ValueError):
            parse_range_header("bytes=0-100,200-300", 1000)


class TestFormatLastModified:
    """Tests for format_last_modified()."""

    def test_formats_datetime(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = format_last_modified(dt)
        assert "Thu, 15 Jan 2026" in result
        assert "GMT" in result
