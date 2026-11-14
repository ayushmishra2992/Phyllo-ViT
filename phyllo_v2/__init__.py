"""Phyllo-ViT 2.0 analysis modules (isolated from the v1.0.1 `phyllo/` package, which is reused, never modified).

Every module returns a *section*: {"status", "reason", ...}. Nothing is ever filled with invented values.
"""
__version__ = "2.0.0"

AVAILABLE = "AVAILABLE"                    # implemented and validated for what it reports
EXPERIMENTAL = "EXPERIMENTAL"              # implemented, heuristic / not validated against labels
USER_INPUT_REQUIRED = "USER_INPUT_REQUIRED"  # needs location / FTIR file / API key from the user
UNAVAILABLE = "UNAVAILABLE"                # tried and failed (network, no station, ...)
NOT_SUPPORTED = "NOT_SUPPORTED"            # no validated data/model exists yet
DISABLED = "DISABLED"                      # deliberately off (cannot be determined from the input)


def section(status, reason, **fields):
    return {"status": status, "reason": reason, **fields}
