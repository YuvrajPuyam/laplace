# Laplace build targets. `make gt-sweeps && make eval` must reproduce the
# README table (spec §9) — those targets land with WS4.

PYTHON ?= python

.PHONY: test test-fast gt-sweeps eval serve mcp live live-gilbreth live-down

# 8800: port 8000 is taken by an unrelated local app on the dev box
serve:
	$(PYTHON) -m uvicorn engine.api:app --port 8800

mcp:
	$(PYTHON) -m engine.mcp_server

# Live decision-twin (Half A) one-command bring-up. `live` is the local toy
# (mock PhysX, no GPU/SSH); `live-gilbreth` drives real Isaac/PhysX over a tunnel.
# Both serve the viewer on :8013 and tear down with `make live-down`.
live:
	bash scripts/live/up.sh

live-gilbreth:
	bash scripts/live/up.sh --gilbreth

live-down:
	bash scripts/live/down.sh

test:
	$(PYTHON) -m pytest -q

test-fast:
	$(PYTHON) -m pytest -q -m "not slow"

gt-sweeps:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m eval.gt_sweep

eval:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m eval.run_eval
