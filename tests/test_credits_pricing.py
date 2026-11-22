"""Unit tests for calculate_credits pricing tiers (src/backend/credits.py).

These are pure-function tests: no Supabase, no Stripe, no app import.
They pin the pricing table's edges, including the duration tiers whose
2.0x branch was unreachable before the 2026-11-22 fix.
"""
from credits import calculate_credits


def test_unknown_generation_type_defaults_to_two():
    assert calculate_credits("totally_unknown_type") == 2


def test_image_hd_multiplier_observable(monkeypatch):
    # PR-Piet (2026-11-22): the 1024x1024 default is already HD (height > 720),
    # so a default-baseline assertion proved nothing. Use a non-HD baseline and
    # a controlled table value that makes the 1.5x multiplier observable.
    import credits

    monkeypatch.setitem(credits.CREDIT_COSTS, credits.GenerationType.SDXL, 4)
    assert calculate_credits("sdxl", width=800, height=600) == 4      # non-HD
    assert calculate_credits("sdxl", width=1920, height=1080) == 6    # int(4 * 1.5)


def test_image_steps_multiplier_over_thirty():
    assert calculate_credits("sdxl", steps=40) == int(calculate_credits("sdxl") * 1.2)


def test_video_duration_tiers_with_real_table():
    # Cost table: VIDEO_I2V_SHORT=5, VIDEO_I2V_MEDIUM=8. Duration drives both
    # the tier selection (>3 s -> medium) and the multiplier (>5 s -> 1.5x,
    # >10 s -> 2.0x). Before the 2026-11-22 fix the 2.0x tier was unreachable
    # (dead elif) and a 12 s video priced the same as a 6 s one (12, not 16).
    assert calculate_credits("video_i2v", width=1024, height=576, duration_seconds=3) == 5
    assert calculate_credits("video_i2v", width=1024, height=576, duration_seconds=6) == 12
    assert calculate_credits("video_i2v", width=1024, height=576, duration_seconds=12) == 16


def test_video_duration_tiers_monotonic():
    short = calculate_credits("video_i2v", width=1024, height=576, duration_seconds=3)
    mid = calculate_credits("video_i2v", width=1024, height=576, duration_seconds=8)
    long_ = calculate_credits("video_i2v", width=1024, height=576, duration_seconds=12)
    assert short < mid < long_


def test_result_floor_is_enforced_via_zero_base(monkeypatch):
    # PR-Piet (2026-11-22): a positive base price can never prove the
    # max(1, ...) clamp - it stays >=1 without it. Force a zero base cost
    # through the cost table and assert the floor produces exactly one.
    import credits

    monkeypatch.setitem(credits.CREDIT_COSTS, credits.GenerationType.SDXL, 0)
    assert calculate_credits("sdxl") == 1


def test_legacy_wan22_alias_pricing_is_a_real_table_entry():
    # Retired 2026-10-01; historical transaction rows must keep pricing
    # identically. PR-Piet (2026-11-22): assert the exact table value so a
    # deleted alias would fall through to the unknown-type default (2) and
    # turn this test red, instead of quietly passing on "> 0".
    assert calculate_credits("wan22_i2v", width=1024, height=576, duration_seconds=3) == 5


def test_hyphenated_type_names_normalise():
    assert calculate_credits("generate-sdxl") == calculate_credits("generate_sdxl")
