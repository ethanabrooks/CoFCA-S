import numpy as np

pairs = [
    # (0.0012086, 5),
    # (0.00094671, 3),
    # (0.0015547, 2),
    # (0.00042464, 3),
    # (0.0015082, 4),
    # (0.019449, 4),
    # (0.0010306, 3),
    # (0.00065722, 5),
    # (0.0017009, 5),
    (0.0021051, 1),
    (0.0021072, 2),
    (0.003915, 1),
    (0.0006596, 5),
    (0.0011849, 2),
    (0.002, 15),
    (5e-05, 20),
    (0.0001, 25),
    (0.0001, 15),
]

x, y = map(np.array, zip(*pairs))
print(x)
print(y)
print(np.polyfit(x, y, 1))
print(np.polyfit(x, y, 2))
a, b, c = np.polyfit(x, y, 2)
for x, y in pairs:
    print(x, y, a * (x) ** 2 + b * x + c, sep=",")
print("low")
for x, y in pairs:
    x = x + 0.002
    print(a * (x) ** 2 + b * x + c)
print("high")
for x, y in pairs:
    x = x + 0.002
    print(a * (x) ** 2 + b * x + c)
