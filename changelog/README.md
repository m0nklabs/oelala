# Fragment-Based Changelog

This directory contains changelog fragments - individual files that describe changes in PRs/commits.

## Why Fragments?

The main `CHANGELOG.md` always causes merge conflicts when multiple PRs are open simultaneously.
By using fragments, each PR adds its own file, and conflicts are impossible.

## How to Add a Fragment

1. Create a file in this directory with naming convention:
   ```
   {PR_NUMBER}-{short-description}.md
   ```

   Examples:
   - `82-progress-tracker-component.md`
   - `83-websocket-progress-events.md`
   - `fix-typo-in-docs.md` (for non-PR changes)

2. Use this template:
   ```markdown
   ### Added
   - **Feature Name**: Brief description
     - Detail 1
     - Detail 2

   ### Fixed
   - Bug fix description

   ### Changed
   - Change description
   ```

3. Use standard changelog sections: `Added`, `Fixed`, `Changed`, `Deprecated`, `Removed`, `Security`

## Merging Fragments

When releasing, run:
```bash
python scripts/merge_changelog.py
```

This will:
1. Combine all fragments into the `[Unreleased]` section of `CHANGELOG.md`
2. Sort entries by section (Added, Fixed, Changed, etc.)
3. Delete the merged fragment files
4. You can then rename `[Unreleased]` to the version number

## CI Integration

The `release.yml` workflow automatically merges fragments before creating a release.
