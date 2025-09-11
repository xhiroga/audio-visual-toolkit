test:
	uv run bash -lc 'sel=$$(rg --files -g "tests/**/*.py" | fzf -m || true); if [ -n "$$sel" ]; then pytest $$sel; else pytest; fi'
