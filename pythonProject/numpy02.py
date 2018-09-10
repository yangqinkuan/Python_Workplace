import numpy as np

a = np.arange(4)
print(a)
b = a
c = a
d = b
a[0] = 11
print(a)
print((b is a))

