#!/usr/bin/env python3
"""
Merge changelog fragments into CHANGELOG.md.

This script combines individual changelog fragment files from the changelog/
directory into the main CHANGELOG.md file under the [Unreleased] section.
"""

import re
import sys
from pathlib import Path

# Section order for changelog entries
SECTION_ORDER = [
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Documentation",
]

DEBUG = False


def log(msg: str) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"🔍 {msg}")


def parse_fragment(content: str) -> dict[str, list[str]]:
    """Parse a fragment file into sections."""
    sections: dict[str, list[str]] = {}
    current_section = None
    current_entries: list[str] = []

    for line in content.split("\n"):
        # Check for section header (### Added, ### Fixed, etc.)
        section_match = re.match(r"^###\s+(\w+)", line)
        if section_match:
            # Save previous section
            if current_section and current_entries:
                sections[current_section] = current_entries
            current_section = section_match.group(1)
            current_entries = []
            log(f"Found section: {current_section}")
        elif current_section and line.strip():
            current_entries.append(line)

    # Save last section
    if current_section and current_entries:
        sections[current_section] = current_entries

    return sections


def merge_sections(all_sections: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Merge multiple section dicts into one."""
    merged: dict[str, list[str]] = {}

    for sections in all_sections:
        for section, entries in sections.items():
            if section not in merged:
                merged[section] = []
            merged[section].extend(entries)

    return merged


def format_unreleased(sections: dict[str, list[str]]) -> str:
    """Format merged sections as markdown."""
    lines = ["## [Unreleased]", ""]

    # Add sections in order
    for section in SECTION_ORDER:
        if section in sections:
            lines.append(f"### {section}")
            lines.extend(sections[section])
            lines.append("")
            del sections[section]

    # Add any remaining sections not in SECTION_ORDER
    for section, entries in sorted(sections.items()):
        lines.append(f"### {section}")
        lines.extend(entries)
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    global DEBUG

    if "--debug" in sys.argv:
        DEBUG = True

    repo_root = Path(__file__).parent.parent
    changelog_dir = repo_root / "changelog"
    changelog_file = repo_root / "CHANGELOG.md"

    # Find all fragment files (exclude README)
    fragments = [
        f for f in changelog_dir.glob("*.md") if f.name.lower() != "readme.md"
    ]

    if not fragments:
        print("✅ No changelog fragments to merge")
        return 0

    print(f"📋 Found {len(fragments)} changelog fragment(s)")

    # Parse all fragments
    all_sections: list[dict[str, list[str]]] = []
    for fragment in fragments:
        log(f"Parsing {fragment.name}")
        content = fragment.read_text()
        sections = parse_fragment(content)
        if sections:
            all_sections.append(sections)

    if not all_sections:
        print("⚠️ No valid changelog entries found in fragments")
        return 0

    # Merge all sections
    merged = merge_sections(all_sections)
    new_unreleased = format_unreleased(merged)

    # Read existing changelog
    if changelog_file.exists():
        existing = changelog_file.read_text()
    else:
        existing = "# Changelog\n\n"

    # Replace or insert [Unreleased] section
    unreleased_pattern = r"## \[Unreleased\].*?(?=\n## \d|\n---|\Z)"
    if re.search(unreleased_pattern, existing, re.DOTALL):
        # Merge with existing unreleased content
        existing_match = re.search(unreleased_pattern, existing, re.DOTALL)
        if existing_match:
            existing_unreleased = existing_match.group(0)
            existing_sections = parse_fragment(existing_unreleased)
            if existing_sections:
                all_sections.insert(0, existing_sections)
                merged = merge_sections(all_sections)
                new_unreleased = format_unreleased(merged)

        new_content = re.sub(
            unreleased_pattern, new_unreleased.rstrip(), existing, count=1, flags=re.DOTALL
        )
    else:
        # Insert after title
        lines = existing.split("\n")
        insert_idx = 1 if lines and lines[0].startswith("#") else 0
        lines.insert(insert_idx, "\n" + new_unreleased)
        new_content = "\n".join(lines)

    # Write updated changelog
    changelog_file.write_text(new_content)
    print(f"✅ Updated {changelog_file}")

    # Delete merged fragments
    if "--dry-run" not in sys.argv:
        for fragment in fragments:
            fragment.unlink()
            log(f"Deleted {fragment.name}")
        print(f"🗑️ Deleted {len(fragments)} fragment(s)")
    else:
        print("🔍 Dry run - fragments not deleted")

    return 0


if __name__ == "__main__":
    sys.exit(main())
