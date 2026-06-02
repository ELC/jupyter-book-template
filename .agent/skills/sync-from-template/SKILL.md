---
name: sync-from-template
description: >-
  Sync upstream infrastructure from jupyter-book-template main into a child
  book project created from this template. Use when the user asks to fetch,
  merge, or apply template updates, sync from upstream, or pull latest
  jupyter-book-template changes.
compatibility: Requires git, uv, and network access to github.com/ELC/jupyter-book-template.
---

# Sync from jupyter-book-template

Child repos created via **Use this template** get a single initial commit with **no shared Git history**. The first sync needs `--allow-unrelated-histories`; later syncs are normal merges. See [GitHub discussion #50012](https://github.com/orgs/community/discussions/50012).

Run all commands from the **child project** root (not this template repo).

## 1. Add upstream remote (once)

```bash
git remote add template https://github.com/ELC/jupyter-book-template.git
git fetch template main
```

If `template` already exists, skip `remote add` and only fetch.

## 2. Merge

**First sync** (child has only an initial commit or never merged template before):

```bash
git merge template/main --allow-unrelated-histories --no-edit
```

**Later syncs:**

```bash
git merge template/main --no-edit
```

Expect the first merge to import the full template commit history; the file diff should still be mostly infrastructure.

## 3. Resolve conflicts

Prefer **template** (`--theirs`) for shared infrastructure:

- `.agent/skills/` (including this skill)
- `.binder/`, `.dockerignore`, `.hidden`
- `.github/workflows/`
- `.pre-commit-config.yaml`
- `AGENTS.md` (unless the child project has diverged intentionally)
- `uv.lock`

Prefer **child** (`--ours`) for project-specific content:

- `book/myst.yml` — `project.title`, `project.github`, `project.copyright`, `project.authors`, `toc` entries for new chapters
- `book/chapters/` — all project chapters (not template demo pages the user removed)
- `pyproject.toml` — `[project].name`, `description`, and content `dependencies`
- `README.md` — badges, URLs, project description
- `book/plugins/ethicalads.mjs` — `book_title` (must match `project.title`)

When the child is still mostly placeholders with no real customization, taking template for all conflicted files is fine.

```bash
# Example: take template version for one file
git checkout --theirs path/to/file
git add path/to/file
```

## 4. Validate

```bash
uv sync --all-groups --locked
uv run poe ci
```

Pre-commit must pass. Book build needs Node/npm locally; CI on GitHub covers that if missing locally.

## 5. Commit and push

Only commit when the user asks. Suggested message:

```
chore: merge jupyter-book-template main

Sync infrastructure updates from the template upstream.
```

Push only when explicitly requested.

## Troubleshooting

**`error: cannot open '.git/FETCH_HEAD': Invalid argument` (Windows)** — a stuck `git` process (often from `git diff` touching `.git/`) holds the lock. Kill lingering `git.exe` processes, delete `.git/FETCH_HEAD`, retry fetch.

**Merge brings entire template history** — expected for template repos; net file changes should still be incremental after the first sync.

**Child not initialized yet** — complete placeholder replacement in `book/myst.yml` and README first; syncing before that is still OK if conflicts are resolved toward template for infra files.
