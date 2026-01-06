## Summary

<!-- Brief description of what this PR does -->

## Related Issue

Fixes #<!-- issue number -->

## Changelog Entry

<!-- REQUIRED: Copy this to CHANGELOG.md under [Unreleased] -->

```markdown
### Added/Changed/Fixed
- Your change description here
```

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 📚 Documentation update
- [ ] 🔧 Refactoring (no functional changes)
- [ ] 🧪 Test coverage improvement

## Checklist

### General
- [ ] My code follows the existing patterns in this repo
- [ ] I have added/updated tests for my changes
- [ ] All new and existing tests pass (`pytest`)
- [ ] Linting passes (`ruff check .`)
- [ ] I have updated documentation if needed
- [ ] **CHANGELOG.md is updated** (required - CI will fail without this!)

### Trading-Specific (if applicable)
- [ ] Paper trading is the default (`dry_run=True` or `paper_mode=True`)
- [ ] No credentials/secrets are hardcoded
- [ ] All order attempts are logged with full details
- [ ] Position limits are enforced
- [ ] Error handling covers network errors, API errors, partial fills

## Testing Instructions

<!-- How can reviewers test this change? -->

```bash
# Example commands to run
pytest tests/test_<module>.py -v
```

## Screenshots (if UI changes)

<!-- Add screenshots here if relevant -->
