"""
Tests for the ``photonics_helper.looks`` module (console helpers).
"""

import pytest
from unittest.mock import patch
from photonics_helper.looks import c_info, c_error


def test_c_info_prints_message():
    """c_info should print a formatted message to the console."""
    with patch("photonics_helper.looks.console.print") as mock_print:
        c_info("test message")
        mock_print.assert_called_once_with("[bold blue]test message[/bold blue]")


def test_c_error_prints_message():
    """c_error should print a formatted error message to the console."""
    with patch("photonics_helper.looks.console.print") as mock_print:
        c_error("error message")
        mock_print.assert_called_once_with("[bold red]:x: error message[/bold red]")
