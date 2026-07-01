"""WS3 — the agent: a thin Claude tool-use loop over the laplace-env tools.

Thin agent, rich engine (spec §2): everything the agent knows about
warehouses, statistics, and budgets lives behind the eight tools. This
package only carries the system-prompt contract, the loop runner, and trace
logging.
"""
