"""A tiny sample program used to demonstrate the call-graph viewer."""

import helpers
from utils import validate, clamp

DATA = [1, 2, 3, 4, 5]


def run(data):
    cleaned = preprocess(data)
    result = process(cleaned)
    report(result)
    return result


def preprocess(data):
    validate(data)
    return helpers.normalize(data)


def process(data):
    total = 0
    for x in data:
        total += helpers.transform(x)
    return clamp(total, 0, 100)


def report(result):
    summarize(result)
    helpers.render(result)


def summarize(result):
    log("result=" + str(result))


def log(msg):
    print(msg)


def debug_dump(state):
    # An "entrypoint": defined here but never called from within the module.
    log(state)
    helpers.render(state)


# module-level code -> this call seeds the graph
run(DATA)
