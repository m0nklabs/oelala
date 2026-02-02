#!/usr/bin/env python3
"""
🥁 Drum Machine vs Human Drummer Analyzer v2
IMPROVED: Analyzes repeating patterns within segments
By Rick Sanchez - now with 47% more interdimensional accuracy
"""

import librosa
import numpy as np
from scipy import stats

def analyze_repeating_patterns(audio_path: str):
    print(f"\n🔬 Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr
    print(f"📊 Duration: {duration:.1f}s, Sample rate: {sr}Hz")

    # === TEMPO & BEAT DETECTION ===
    print("\n🎵 Detecting tempo and beats...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"   Tempo: {tempo:.1f} BPM")
    print(f"   Detected {len(beat_times)} beats")

    # === ONSET DETECTION ===
    print("\n🥁 Detecting drum onsets...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_strengths = onset_env[onset_frames]
    print(f"   Found {len(onset_times)} onsets")

    # === SEGMENT INTO BARS (4 beats = 1 bar) ===
    beats_per_bar = 4
    bar_duration = (60 / tempo) * beats_per_bar
    print(f"\n📐 Bar duration: {bar_duration:.3f}s ({beats_per_bar} beats)")

    # Group onsets by bar
    num_bars = int(duration / bar_duration)
    print(f"   Total bars: {num_bars}")

    # === ANALYZE EACH BAR'S PATTERN ===
    bar_patterns = []

    for bar_idx in range(num_bars):
        bar_start = bar_idx * bar_duration
        bar_end = bar_start + bar_duration

        # Get onsets in this bar
        mask = (onset_times >= bar_start) & (onset_times < bar_end)
        bar_onsets = onset_times[mask] - bar_start  # Relative to bar start
        bar_velocities = onset_strengths[mask]

        if len(bar_onsets) >= 2:
            # Normalize to 0-1 within bar
            bar_onsets_norm = bar_onsets / bar_duration
            bar_patterns.append({
                'bar_idx': bar_idx,
                'onsets': bar_onsets_norm,
                'onset_times_ms': bar_onsets * 1000,
                'velocities': bar_velocities,
                'num_hits': len(bar_onsets)
            })

    print(f"   Usable bars (≥2 hits): {len(bar_patterns)}")

    # === FIND SIMILAR BARS (same number of hits = likely same pattern) ===
    print("\n🔍 Finding repeating patterns...")

    # Group bars by number of hits
    from collections import defaultdict
    hit_groups = defaultdict(list)
    for bp in bar_patterns:
        hit_groups[bp['num_hits']].append(bp)

    # Analyze groups with enough repetitions
    pattern_analyses = []

    for num_hits, bars in hit_groups.items():
        if len(bars) >= 4 and num_hits >= 3:  # Need at least 4 repetitions, 3+ hits
            print(f"\n   Pattern with {num_hits} hits/bar ({len(bars)} occurrences):")

            # Compare timing between repetitions of same pattern
            timing_variations = []
            velocity_variations = []

            for i, bar in enumerate(bars):
                # For each hit position, calculate deviation from mean position
                if i == 0:
                    # First bar is reference
                    ref_onsets = bar['onsets']
                    ref_velocities = bar['velocities']
                else:
                    # Compare to first bar (same pattern)
                    if len(bar['onsets']) == len(ref_onsets):
                        # Timing difference for each hit
                        timing_diff = np.abs(bar['onsets'] - ref_onsets) * bar_duration * 1000
                        timing_variations.extend(timing_diff)

                        # Velocity difference
                        vel_diff = np.abs(bar['velocities'] - ref_velocities)
                        velocity_variations.extend(vel_diff)

            if timing_variations:
                avg_timing_var = np.mean(timing_variations)
                max_timing_var = np.max(timing_variations)
                std_timing_var = np.std(timing_variations)

                avg_vel_var = np.mean(velocity_variations) if velocity_variations else 0

                print(f"      Timing variation (same hit, different bars):")
                print(f"         Mean: {avg_timing_var:.2f}ms")
                print(f"         Std:  {std_timing_var:.2f}ms")
                print(f"         Max:  {max_timing_var:.2f}ms")
                print(f"      Velocity variation: {avg_vel_var:.2f}")

                pattern_analyses.append({
                    'num_hits': num_hits,
                    'repetitions': len(bars),
                    'timing_mean': avg_timing_var,
                    'timing_std': std_timing_var,
                    'timing_max': max_timing_var,
                    'velocity_var': avg_vel_var
                })

    # === ALTERNATIVE: 16th note grid analysis ===
    print("\n📏 16th Note Grid Analysis:")
    sixteenth_duration = (60 / tempo) / 4  # Duration of 1 sixteenth note

    # Quantize all onsets to nearest 16th
    grid_deviations = []
    for onset in onset_times:
        nearest_16th = round(onset / sixteenth_duration) * sixteenth_duration
        deviation_ms = abs(onset - nearest_16th) * 1000
        grid_deviations.append(deviation_ms)

    grid_mean = np.mean(grid_deviations)
    grid_std = np.std(grid_deviations)
    grid_median = np.median(grid_deviations)

    print(f"   Mean deviation from 16th grid: {grid_mean:.2f}ms")
    print(f"   Median deviation: {grid_median:.2f}ms")
    print(f"   Std deviation: {grid_std:.2f}ms")

    # === INTER-ONSET INTERVAL CONSISTENCY FOR SAME SUBDIVISIONS ===
    print("\n🎯 IOI Consistency (same subdivisions):")
    ioi = np.diff(onset_times) * 1000

    # Find IOIs that are close to common subdivisions
    eighth_note_ms = (60 / tempo) / 2 * 1000
    sixteenth_ms = (60 / tempo) / 4 * 1000

    # Group IOIs by target subdivision
    eighth_iois = ioi[(ioi > eighth_note_ms * 0.7) & (ioi < eighth_note_ms * 1.3)]
    sixteenth_iois = ioi[(ioi > sixteenth_ms * 0.7) & (ioi < sixteenth_ms * 1.3)]

    if len(eighth_iois) > 5:
        eighth_std = np.std(eighth_iois)
        print(f"   8th note IOIs ({len(eighth_iois)} found): std = {eighth_std:.2f}ms")

    if len(sixteenth_iois) > 5:
        sixteenth_std = np.std(sixteenth_iois)
        print(f"   16th note IOIs ({len(sixteenth_iois)} found): std = {sixteenth_std:.2f}ms")

    # === FINAL VERDICT ===
    print("\n" + "="*70)
    print("🎯 IMPROVED VERDICT (Pattern-Based Analysis)")
    print("="*70)

    machine_score = 0
    human_score = 0

    # Pattern timing consistency
    if pattern_analyses:
        avg_pattern_timing = np.mean([p['timing_mean'] for p in pattern_analyses])
        print(f"\n   📊 Average timing variation in repeating patterns: {avg_pattern_timing:.2f}ms")

        if avg_pattern_timing < 3:
            machine_score += 4
            print("      ⚡ EXTREMELY consistent (<3ms) → Almost certainly drum machine")
        elif avg_pattern_timing < 8:
            machine_score += 2
            print("      ⚡ Very consistent (3-8ms) → Likely drum machine or heavy quantize")
        elif avg_pattern_timing < 15:
            human_score += 1
            print("      🤔 Moderate variation (8-15ms) → Could be either")
        else:
            human_score += 3
            print("      🎸 High variation (>15ms) → Likely human")

    # Grid deviation
    print(f"\n   📊 16th note grid deviation: {grid_mean:.2f}ms (median: {grid_median:.2f}ms)")
    if grid_median < 5:
        machine_score += 2
        print("      ⚡ Locked to grid → Drum machine or quantized")
    elif grid_median < 12:
        print("      🤔 Close to grid but with swing → Could be either")
    else:
        human_score += 2
        print("      🎸 Significant grid deviation → Human feel")

    # IOI consistency
    if len(sixteenth_iois) > 5:
        print(f"\n   📊 16th note IOI consistency: {sixteenth_std:.2f}ms std")
        if sixteenth_std < 5:
            machine_score += 2
            print("      ⚡ Perfect subdivision timing → Drum machine")
        elif sixteenth_std < 10:
            machine_score += 1
            print("      ⚡ Very consistent subdivisions → Likely programmed")
        else:
            human_score += 1
            print("      🎸 Natural subdivision variation → Human")

    print("\n" + "-"*70)
    total = machine_score + human_score
    if total > 0:
        machine_pct = machine_score / total * 100
        human_pct = human_score / total * 100
    else:
        machine_pct = human_pct = 50

    print(f"   🤖 Drum Machine likelihood: {machine_pct:.0f}%")
    print(f"   🧑 Human Drummer likelihood: {human_pct:.0f}%")

    if machine_score > human_score + 2:
        print("\n   📌 CONCLUSION: Almost certainly DRUM MACHINE or heavily quantized")
    elif machine_score > human_score:
        print("\n   📌 CONCLUSION: Probably DRUM MACHINE or electronic/quantized")
    elif human_score > machine_score + 2:
        print("\n   📌 CONCLUSION: Almost certainly HUMAN DRUMMER")
    elif human_score > machine_score:
        print("\n   📌 CONCLUSION: Probably HUMAN with natural feel")
    else:
        print("\n   📌 CONCLUSION: Inconclusive - hybrid or lightly processed")

    print("="*70)

if __name__ == "__main__":
    analyze_repeating_patterns("/home/flip/oelala/scratch/drum_analysis.wav")
