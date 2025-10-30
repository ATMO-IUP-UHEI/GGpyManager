"""Logging configuration utilities."""

import logging


def set_logger(level: str = "INFO") -> None:
    """Set up the logging configuration.

    Parameters
    ----------
    level : str, optional
        Logging level. Default is 'INFO'.
    """
    logging.basicConfig(
        level=getattr(
            logging, level.upper(), logging.INFO
        ),  # Use the function argument
        format="%(levelname)s: %(message)s",
        force=True,  # Force reset of logging settings
    )
    logging.info("Logger set up.")
