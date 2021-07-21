import numpy as np

a = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
b = np.array([2, 1, 3])
# print(a[:, b])
# print(np.select(a, b))

print(b[:, np.newaxis])
# np.

print(range(len(a)))
a_new = a[np.array(range(len(a)))[:, np.newaxis], b]
# a_new = a[np.ix_(range(len(a)), b)]
print(a_new)
print(a[range(a.shape[0]), b])
