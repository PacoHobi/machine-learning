import numpy as np


def averages(values):
    averages_ = []

    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
        avg = total / count
        averages_.append(avg)

    return averages_