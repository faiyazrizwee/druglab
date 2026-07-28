"""
DrugLab Logging System.

Provides structured, Rich-formatted console output and file logging
for all DrugLab modules. Every module should obtain its logger via
setup_logger() to ensure consistent formatting and dual output
(console + file).

Usage:
    from utils.logger import setup_logger, log_step, log_summary
    logger = setup_logger("protein_preparation", log_dir="output/logs")
    logger.info("Starting protein preparation")
    log_step(logger, "remove_waters", "completed", {"count": 127})
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ──────────────────────────────────────────────────────────────
# Module-level Rich console (shared across the application)
# ──────────────────────────────────────────────────────────────

console = Console()

# Track active step timers: {step_name: start_time}
_step_timers: dict[str, float] = {}


# ──────────────────────────────────────────────────────────────
# Logger Setup
# ──────────────────────────────────────────────────────────────

def setup_logger(
    name: str,
    log_dir: str = "output/logs",
    level: str = "INFO",
    verbose: bool = False,
) -> logging.Logger:
    """
    Create a named logger with Rich console handler and file handler.

    Args:
        name: Logger name (typically the module name, e.g. "protein_preparation").
        log_dir: Directory for log files. Created if it doesn't exist.
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        verbose: If True, overrides level to DEBUG.

    Returns:
        Configured logging.Logger instance.
    """
    if verbose:
        level = "DEBUG"

    logger = logging.getLogger(f"druglab.{name}")

    # Avoid adding duplicate handlers if setup_logger is called twice
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # ── Rich Console Handler ──
    rich_handler = RichHandler(
        console=console,
        show_path=False,
        show_time=True,
        rich_tracebacks=True,
        tracebacks_show_locals=verbose,
        markup=True,
    )
    rich_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    rich_format = logging.Formatter("%(message)s")
    rich_handler.setFormatter(rich_format)
    logger.addHandler(rich_handler)

    # ── File Handler ──
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always capture everything to file
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    logger.debug(f"Logger initialized: {name} (level={level}, log_file={log_file})")
    return logger


# ──────────────────────────────────────────────────────────────
# Structured Step Logging
# ──────────────────────────────────────────────────────────────

def log_step(
    logger: logging.Logger,
    step_name: str,
    status: str,
    details: dict | None = None,
) -> None:
    """
    Log a structured workflow step with status and optional details.

    Produces Rich panel output to console and structured text to file.

    Args:
        logger: The logger instance.
        step_name: Human-readable step name (e.g. "Protein Preparation").
        status: Step status — "started", "completed", "failed", "skipped".
        details: Optional dict of key-value pairs to display.
    """
    status_lower = status.lower()

    # Start timer on "started"
    if status_lower == "started":
        _step_timers[step_name] = time.time()
        logger.info(f"[STEP] {step_name} | STATUS: {status}")
        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
        return

    # Calculate duration on completion/failure
    duration_str = ""
    if step_name in _step_timers:
        elapsed = time.time() - _step_timers.pop(step_name)
        if elapsed < 60:
            duration_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            duration_str = f"{elapsed / 60:.1f}m"
        else:
            duration_str = f"{elapsed / 3600:.1f}h"

    # File log
    detail_str = ""
    if details:
        detail_str = " | " + " | ".join(f"{k}={v}" for k, v in details.items())
    duration_part = f" | duration: {duration_str}" if duration_str else ""
    logger.info(f"[STEP] {step_name} | STATUS: {status}{duration_part}{detail_str}")

    # Rich console panel
    if status_lower in ("completed", "failed", "skipped"):
        _render_step_panel(step_name, status, details, duration_str)


def _render_step_panel(
    step_name: str,
    status: str,
    details: dict | None,
    duration_str: str,
) -> None:
    """Render a Rich panel for a completed/failed step."""
    status_lower = status.lower()

    if status_lower == "completed":
        border_style = "green"
        status_icon = "✓"
    elif status_lower == "failed":
        border_style = "red"
        status_icon = "✗"
    elif status_lower == "skipped":
        border_style = "yellow"
        status_icon = "⊘"
    else:
        border_style = "blue"
        status_icon = "●"

    # Build panel content
    lines = []
    if details:
        for key, value in details.items():
            lines.append(f"  {key}: {value}")

    footer = f"{status_icon} {status}"
    if duration_str:
        footer += f" in {duration_str}"

    content = "\n".join(lines) if lines else ""

    panel = Panel(
        content,
        title=f"[bold]{step_name}[/bold]",
        subtitle=footer,
        border_style=border_style,
        padding=(0, 2),
    )
    console.print(panel)


# ──────────────────────────────────────────────────────────────
# Summary Table
# ──────────────────────────────────────────────────────────────

def log_summary(logger: logging.Logger, results: dict) -> None:
    """
    Print a Rich summary table of workflow results.

    Args:
        logger: The logger instance.
        results: Dict mapping stage names to result dicts. Each result dict
                 should contain at minimum 'status' and optionally
                 'duration', 'input_count', 'output_count', etc.
    """
    table = Table(
        title="[bold]DrugLab Workflow Summary[/bold]",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
        pad_edge=True,
    )
    table.add_column("Stage", style="bold", min_width=25)
    table.add_column("Status", justify="center", min_width=10)
    table.add_column("Input", justify="right", min_width=8)
    table.add_column("Output", justify="right", min_width=8)
    table.add_column("Duration", justify="right", min_width=10)

    for stage_name, stage_result in results.items():
        status = stage_result.get("status", "unknown")
        if status == "completed":
            status_text = Text("✓ done", style="green")
        elif status == "failed":
            status_text = Text("✗ fail", style="red")
        elif status == "skipped":
            status_text = Text("⊘ skip", style="yellow")
        else:
            status_text = Text(status, style="dim")

        input_count = str(stage_result.get("input_count", "—"))
        output_count = str(stage_result.get("output_count", "—"))
        duration = stage_result.get("duration", "—")

        table.add_row(stage_name, status_text, input_count, output_count, str(duration))

    console.print()
    console.print(table)
    console.print()

    # Also log to file in plain text
    logger.info("=== Workflow Summary ===")
    for stage_name, stage_result in results.items():
        status = stage_result.get("status", "unknown")
        logger.info(
            f"  {stage_name}: {status} "
            f"(in={stage_result.get('input_count', '?')}, "
            f"out={stage_result.get('output_count', '?')}, "
            f"time={stage_result.get('duration', '?')})"
        )


# ──────────────────────────────────────────────────────────────
# Convenience: Print a DrugLab banner
# ──────────────────────────────────────────────────────────────

def print_banner() -> None:
    """Print the DrugLab startup banner."""
    banner_text = Text()
    banner_text.append("Drug", style="bold cyan")
    banner_text.append("Lab", style="bold green")

    console.print()
    console.print(
        Panel(
            banner_text,
            subtitle="Computational Drug Discovery Platform",
            border_style="bright_blue",
            padding=(1, 4),
        )
    )
    console.print()
