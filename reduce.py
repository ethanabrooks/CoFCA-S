#! /usr/bin/env python

import functools


def f(hxs, t):
    *hxs, hx = hxs
    return [*hxs, hx, hx + 1]


print(functools.reduce(f, range(5), [0]))
