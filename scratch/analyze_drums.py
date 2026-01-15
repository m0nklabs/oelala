#!/usr/bin/env python3
"""
🥁 Drum Machine vs Human Drummer Analyzer
By Rick Sanchez - the smartest being in the multiverse
"""

import librosa
import numpy as np
from pathlib import Path

def analyze_drums(audio_path: str):
    print(f"\n🔬 Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr
    print(f"📊 Duration: {duration:.1f}s, Sample rate: {sr}Hz")
    
    # === ONSET DETECTION (find drum hits) ===
    print("\n🥁 Detecting drum onsets...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, 
        sr=sr,
        backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"   Found {len(onset_times)} onsets")
    
    if len(onset_times) < 10:
        print("❌ Not enough drum hits detected for analysis")
        return
    
    # === TIMING ANALYSIS ===
    print("\n⏱️  Timing Analysis (the key indicator):")
    ioi = np.diff(onset_times) * 1000  # Inter-onset intervals in ms
    
    # Filter out very long gaps (likely different sections)
    ioi_filtered = ioi[ioi < 1000]  # Only consider < 1 second gaps
    
    timing_std = np.std(ioi_filtered)
    timing_mean = np.mean(ioi_filtered)
    timing_cv = timing_std / timing_mean * 100  # Coefficient of variation
    
    print(f"   Mean IOI: {timing_mean:.1f}ms (~{60000/timing_mean:.0f} BPM)")
    print(f"   Std Dev:  {timing_std:.2f}ms")
    print(f"   CV:       {timing_cv:.2f}%")
    
    # === TEMPO ESTIMATION ===
    print("\n🎵 Tempo Analysis:")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # Handle both old and new librosa API
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0
    print(f"   Estimated tempo: {tempo:.1f} BPM")
    
    # === MICROTIMING DEVIATION FROM GRID ===
    print("\n📐 Grid Deviation Analysis:")
    if tempo > 0:
        beat_duration_ms = 60000 / tempo
        # Calculate deviation from perfect grid
        grid_deviations = []
        for ioi_val in ioi_filtered:
            # Find closest grid subdivision (1, 1/2, 1/4 beat)
            for div in [1, 0.5, 0.25, 0.125]:
                expected = beat_duration_ms * div
                if abs(ioi_val - expected) < expected * 0.3:
                    deviation = abs(ioi_val - expected)
                    grid_deviations.append(deviation)
                    break
        
        if grid_deviations:
            avg_deviation = np.mean(grid_deviations)
            max_deviation = np.max(grid_deviations)
            print(f"   Avg grid deviation: {avg_deviation:.2f}ms")
            print(f"   Max grid deviation: {max_deviation:.2f}ms")
    
    # === VELOCITY/DYNAMICS ANALYSIS ===
    print("\n🔊 Dynamics Analysis:")
    onset_strengths = onset_env[onset_frames]
    velocity_std = np.std(onset_strengths)
    velocity_cv = velocity_std / np.mean(onset_strengths) * 100
    print(f"   Velocity variation (CV): {velocity_cv:.1f}%")
    
    # === SPECTRAL CONSISTENCY (same samples = low variation) ===
    print("\n🌈 Spectral Consistency:")
    # Get spectral centroids at onset points
    spec_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_frames = np.minimum(onset_frames, len(spec_centroids) - 1)
    onset_centroids = spec_centroids[centroid_frames]
    spectral_cv = np.std(onset_centroids) / np.mean(onset_centroids) * 100
    print(f"   Spectral centroid CV: {spectral_cv:.1f}%")
    
    # === VERDICT ===
    print("\n" + "="*60)
    print("🎯 VERDICT")
    print("="*60)
    
    # Scoring system
    machine_score = 0
    human_score = 0
    
    # Timing precision
    if timing_std < 5:
        machine_score += 3
        print("   ⚡ Timing: EXTREMELY precise (<5ms) → Drum machine")
    elif timing_std < 10:
        machine_score += 2
        print("   ⚡ Timing: Very precise (5-10ms) → Likely drum machine/heavy quantize")
    elif timing_std < 20:
        human_score += 1
        print("   🎸 Timing: Natural variation (10-20ms) → Could be human")
    else:
        human_score += 2
        print("   🎸 Timing: High variation (>20ms) → Likely human")
    
    # Grid deviation
    if 'avg_deviation' in dir():
        if avg_deviation < 3:
            machine_score += 2
            print("   ⚡ Grid: Locked to grid (<3ms avg) → Drum machine")
        elif avg_deviation < 8:
            machine_score += 1
            print("   ⚡ Grid: Close to grid (3-8ms) → Possibly quantized")
        else:
            human_score += 1
            print("   🎸 Grid: Natural swing (>8ms) → Human feel")
    
    # Velocity
    if velocity_cv < 10:
        machine_score += 1
        print("   ⚡ Dynamics: Very consistent → Programmed")
    elif velocity_cv < 25:
        print("   🤔 Dynamics: Moderate variation → Could be either")
    else:
        human_score += 1
        print("   🎸 Dynamics: High variation → Human touch")
    
    # Spectral consistency (identical samples)
    if spectral_cv < 15:
        machine_score += 1
        print("   ⚡ Timbre: Very consistent samples → Drum machine")
    else:
        human_score += 1
        print("   🎸 Timbre: Natural variation → Acoustic/Human")
    
    print()
    total = machine_score + human_score
    if total > 0:
        machine_pct = machine_score / total * 100
        human_pct = human_score / total * 100
    else:
        machine_pct = human_pct = 50
    
    print(f"   🤖 Drum Machine likelihood: {machine_pct:.0f}%")
    print(f"   🧑 Human Drummer likelihood: {human_pct:.0f}%")
    
    if machine_score > human_score + 2:
        print("\n   📌 CONCLUSION: Almost certainly a DRUM MACHINE or heavily quantized")
    elif machine_score > human_score:
        print("\n   📌 CONCLUSION: Probably DRUM MACHINE or electronic drums")
    elif human_score > machine_score + 2:
        print("\n   📌 CONCLUSION: Almost certainly a HUMAN DRUMMER")
    elif human_score > machine_score:
        print("\n   📌 CONCLUSION: Probably HUMAN or lightly quantized")
    else:
        print("\n   📌 CONCLUSION: Inconclusive - could be either or hybrid")
    
    print("="*60)

if __name__ == "__main__":
    analyze_drums("/home/flip/oelala/scratch/drum_analysis.wav")
