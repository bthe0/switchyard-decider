# Contributing

Issues and pull requests are welcome.

1. Install the development dependencies: `pip install -e ".[dev]"`.
2. Keep changes small and focused. Pure scoring logic belongs in numpy functions so it can be tested without torch.
3. Run `ruff check .` and `pytest -q tests` before opening a pull request.
4. Do not change the logit order, the gold-logit sign convention, the `calib.npz` keys or the HTTP response format
   without a major version bump; see [CLAUDE.md](CLAUDE.md) for the invariants.
5. Never commit datasets, weights, credentials or API keys.
6. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

By contributing you agree that your contributions are licensed under the Apache-2.0 license.
