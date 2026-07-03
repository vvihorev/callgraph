"""Helper functions imported by main.py."""


def normalize(data):
    return [strip(x) for x in data]


def strip(x):
    return x


def transform(x):
    return scale(x) + offset()


def scale(x):
    return x * 2


def offset():
    return 1


def render(x):
    print(x)
