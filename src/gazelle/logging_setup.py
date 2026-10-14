import logging
import warnings


def quiet_startup() -> None:
    """Drop third-party startup noise that is not actionable.

    Targets specific messages rather than whole loggers, so genuinely useful
    lines — "GPU available: True (mps), used: True" — still print.
    """
    warnings.filterwarnings(
        "ignore", message=r"`isinstance\(treespec, LeafSpec\)` is deprecated"
    )

    class _DropLitLoggerTip(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "litlogger" not in record.getMessage().lower()

    logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
        _DropLitLoggerTip()
    )
