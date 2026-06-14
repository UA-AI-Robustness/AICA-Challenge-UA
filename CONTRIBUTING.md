# Contributing Guide

## Branch Strategy

- `main` — always working, always passes tests. Never commit directly.
- `dev` — integration branch. All feature branches merge here first.
- `feature/<short-description>` — your working branches.

Examples:
- `feature/task-assignment`
- `feature/drone-handoff-coordination`
- `feature/astar-path-planning`

## Workflow

1. Pull latest `dev`:
```bash
   git checkout dev
   git pull origin dev
```
2. Create your feature branch:
```bash
   git checkout -b feature/your-feature-name
```
3. Make changes, commit often.
4. Push your branch:
```bash
   git push -u origin feature/your-feature-name
```
5. Open a Pull Request to `dev` on GitHub.
6. Get one teammate to review.
7. Merge into `dev`. When `dev` is stable, merge to `main`.

## Commit Messages

Format: `<type>: <short description>`

Types:
- `feat`: new feature
- `fix`: bug fix
- `refactor`: code restructuring without behavior change
- `docs`: documentation only
- `test`: adding tests
- `chore`: maintenance

Examples:
- `feat: implement A* path planning for QCar2`
- `fix: correct cross-track error sign in Stanley controller`
- `docs: add scoring formula to overview`

Keep messages under 72 characters for the first line.

## Code Style

- Follow PEP 8 for Python.
- Use descriptive variable names. `delivery_assignment` not `da`.
- Add docstrings to all functions.
- Comment *why*, not *what*.

## Pull Requests

- Title: same format as commit messages.
- Description: what changed, why, how to test.
- Mention teammates with `@username` for review.

## Don't Commit

- Anything matched by `.gitignore`
- Personal scratch code (use the `scratch/` folder — it's gitignored)
- Large binary files (>50MB) without discussion
- Secrets, API keys, or license files