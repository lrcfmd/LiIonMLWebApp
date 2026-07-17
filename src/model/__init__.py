import traceback

from infrastructure.logging import newLogger
from model.model import LiIonModel

_logger = newLogger("model")

# Wire the concrete model for the runner. The runner imports `handler`
# from here and calls handler.process(mode, values, files, output_dir, ...).
try:
    handler = LiIonModel()
except Exception:
    # Log the full error through structlog (stdout) so the LMDS server
    # captures it. Python's default traceback goes to stderr which may be
    # truncated or dropped by the server's log capture.
    _logger.error("model initialization failed", error=traceback.format_exc())
    raise