# PRP: [Feature Name]

## 📖 Context Reading

**Important**: Please read CLAUDE.md completely first and if necessary relevant docs from ai-doc/ and CHANGELOG.md

## 🎯 Goal

**What should be built**: [Concrete description]
**Success definition**: [How do I know it's complete]

## 🧑‍💻 User & Use Case

**Target audience**: [Who uses this feature]
**Main application**: [When/how is it used]
**Problem solved**: [What problem does this solve]

## ✅ Success Criteria

- [ ] [Measurable success criteria]
- [ ] [Tests pass]
- [ ] [Feature works as expected]

## 🧩 Context & References

### Important Files (if available)

- `CLAUDE.md` - Development guidelines (ALWAYS read)
- `ai-doc/[RELEVANT].md` - Specific documentation
- `CHANGELOG.md` - Understand recent changes

### Similar Features in Code

[References to similar implementations in codebase]

### External References

[Relevant documentation, tutorials, etc.]

## 🏗️ Technical Details

### Main Components

[What needs to be implemented/changed]

### Files to Edit/Create

[Specific file paths]

### Dependencies

[External libraries, internal services]

## 🧪 Validation

### Tests

```bash
# Basic tests
uv run pytest src/ -v
uv run ruff check src/
uv run mypy src/
```

### Manual Tests

[Specific commands for testing]

### Integration Tests

[If needed, how do I integrate with other services]

## 🚫 Anti-Patterns to Avoid

- ❌ Don't ignore existing patterns
- ❌ Don't skip tests
- ❌ Don't duplicate code without good reason
- ❌ Don't hardcode what should be configurable

## 📝 Implementation Notes

[Additional hints, pitfalls, etc.]

---

**Template Version**: Simple v1.0 for University Projects
