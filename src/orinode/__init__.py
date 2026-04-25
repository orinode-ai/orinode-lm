"""Orinode-LM: Nigerian Speech-LLM — English / Hausa / Igbo / Yoruba / Pidgin."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv

load_dotenv()

try:
    __version__: str = version("orinode-lm")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
