# Jupyter Book Template

**jupyter-book-template** — Jupyter Book 2 template for MyST notebooks and GitHub Pages deployment.

[![Jupyter Book Badge](https://img.shields.io/badge/jupyter-book-orange?style=for-the-badge&logo=data:image/png%3Bbase64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAZCAMAAAAVHr4VAAAAXVBMVEX////v7+/zdybv7+/zdybv7+/zdybv7+/zdybv7+/zdybv7+/zdybv7+/zdybv7+/zdybv7+/zdybv7+/v7+/zdybv7+/zdybv7+/v7+/zdybv7+/zdybv7+/zdyaSmqV2AAAAHXRSTlMAEBAgIDAwQEBQUGBgcHCAgJCQoLCwwMDQ4ODw8MDkUIUAAADJSURBVHjaddAFkgNBCAXQP+7uAvc/5tLFVseYF8crUB0560r/5gwvjYYm8gq8QJoyIJNwlnUH0WEnART6YSezV6c5tjOTaoKdfGXtnclFlEBEXVd8JzG4pa/LDql9Jff/ZCC/h2zSqF5bzf4vqkgNwEzeClUd8uMadLE6OnhBFsES5niQh2BOYUqZsfGdmrmbN+TMvPROHUOkde8sEs6Bnr0tDDf2Roj6fmVfubuGyttejCeLc+xFm+NLuLnJeFAyl3gS932MF/wBoukfUcwI05kAAAAASUVORK5CYII=)](https://elc.github.io/jupyter-book-template/)
[![badge](https://img.shields.io/badge/launch-binder-579aca.svg?style=for-the-badge&logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/elc/jupyter-book-template/HEAD)
[![badge](https://img.shields.io/badge/Open-In%20Colab-579aca?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/ELC/jupyter-book-template/blob/main/book/chapters/00_how_to_use.ipynb)

This repository is a **manual** template: you edit some settings by hand after forking or copying it.

Check the final result [online](https://elc.github.io/jupyter-book-template/)

## What ships in `src/`

Beyond the book template, this repository includes a **typed ML demo library** (~500 LOC) that chapters import directly after `uv sync`:

| Package | Role |
|---------|------|
| `core` | Pandera schemas, `Settings`, sklearn `RegressorMixin` / `TransformerMixin`, `prepare_split` |
| `simulation` | Synthetic regression dataset |
| `prediction` | Random forest + MAPIE split conformal intervals |
| `analysis` | Case bootstrap confidence intervals (`tqdm` progress) |
| `evaluation` | Bootstrap CIs for regression metrics; MAPIE width/MWI for intervals |
| `visualization` | Altair plots with shared theme constants |

See [`book/chapters/03_using_src_code.ipynb`](book/chapters/03_using_src_code.ipynb) for the end-to-end notebook walkthrough (data → train → bootstrap CI → conformal PI → metrics → plots).

Developer commands:

```bash
uv sync --all-groups
uv run poe test          # 100% coverage gate
uv run poe ci            # pre-commit + tests + book build
```

## Testing a branch before merging

Use this when a pull request changes Binder (`.binder/`), Python dependencies, or in-page execution—and you want to verify the environment before merging to **`main`**.

1. **Push the branch** to GitHub (MyBinder builds from the remote ref).
2. **Point Binder at the branch** in `book/myst.yml`:

   ```yaml
   project:
     jupyter:
       binder:
         ref: your-branch-name
   ```

   (With no `ref`, MyST uses the repository default branch.)

3. **Test on MyBinder** (builds from `.binder/Dockerfile` on GitHub, not GitHub Actions):
   - Branch URL: `https://mybinder.org/v2/gh/ELC/jupyter-book-template/<branch>`
   - Prefer a **commit SHA** URL after each push so MyBinder does not reuse an older cached image:
     `https://mybinder.org/v2/gh/ELC/jupyter-book-template/<commit-sha>`
   - MyBinder [caches builds per commit](https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html); if the log still shows `RUN uv sync` on line 9, you are viewing a stale Dockerfile—open a fresh launch with the latest SHA.
   - Confirm the remote file matches:
     `https://github.com/ELC/jupyter-book-template/blob/<branch>/.binder/Dockerfile` (should include `UV_PROJECT_ENVIRONMENT` and `rm -rf .venv`).
4. **Update launch links** for the same branch (README badge, Colab, `book/myst.yml` `binder.ref`).
5. **Build the image locally** (optional; matches CI `check-docker`):

   ```bash
   uv run poe build-docker
   docker run --rm -p 8888:8888 jupyter-book-template:binder
   ```

6. Confirm notebooks run with the expected kernel on MyBinder or locally.
7. **Before merging the PR**, revert temporary branch pins: remove `project.jupyter.binder.ref` (or set it back to `main`), and restore README badge URLs to **`main`** or **`HEAD`**. Do not merge with a feature branch still configured in `myst.yml`.

## Getting Started

### Step 1: Fork this Repository

This is a *template* repository, so you won't use it directly, you will use it a a template for new repositories. Github only allows to use templates you own so you will have to first fork it and follow these steps:

- [ ] Fork the repository.
- [ ] Enable **Template Repository** in Settings > Options > Template Repository.
- [ ] Enable Github Actions if disable.
- [ ] Enable GitHub Pages with source **GitHub Actions** in Settings → Pages.
- [ ] Clone the repo in your local PC.

### Step 2: Use the Template

Now, when creating a new repository, select this template from the dropdown.

### Step 3: Connect to upstream (do this before customizing)

GitHub template repos do not copy commit history—they create a fresh initial commit ([details](https://github.com/orgs/community/discussions/50012)). Link your new repo to this template and merge **before** you edit placeholders or add chapters. That first merge is usually clean; merging later, after customization, tends to produce many conflicts on shared infrastructure files.

Clone the new repo, then run:

```bash
git remote add template https://github.com/ELC/jupyter-book-template.git
git fetch template main
git merge template/main --allow-unrelated-histories --no-edit
git push origin main
```

For later template updates: `git fetch template main && git merge template/main --no-edit`. See [`.agent/skills/sync-from-template/SKILL.md`](.agent/skills/sync-from-template/SKILL.md) for conflict rules and troubleshooting.

### Step 4: Customize

Open the folder with your favourite IDE / Editor and look for the term `REPLACE WITH` and replace the values with your custom information and remove the brackets. Examples for each are given below:

- **Book Title** -> Book Template
- **Current Year** -> 2021
- **Project URL** -> https://elc.github.com/jupyter-book-template
- **Repository URL** -> https://github.com/elc/jupyter-book-template
- **Binder URL** -> https://mybinder.org/v2/gh/elc/jupyter-book-template/main
- **Google Analytics Measurement ID** -> set `site.options.analytics_google` in `book/myst.yml` (see [MyST analytics](https://mystmd.org/guide/analytics#google-analytics))

Then, some final TODOs:

- [ ] Add the public link to the Github Page of the Repo.
- [ ] Update Links to relevant Badges in this document.
- [ ] Check all available features in the Introduction chapter that comes with the book.
- [ ] Delete this README and update it with project relevant information.
