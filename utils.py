"""Utility functions imported by main.py."""


def validate(data):
    if not data:
        raise ValueError("empty data")
    return True


def clamp(value, lo, hi):
    return max(lo, min(hi, value))
