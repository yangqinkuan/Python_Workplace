import numpy as np


pop = np.random.randint(0, 2, (1, 10)).repeat(100, axis=0)
dot = 2 ** np.arange(10)[::-1]
translateDNA = pop.dot(2 ** np.arange(10)[::-1])
print(pop)
print(dot)
print(translateDNA)
