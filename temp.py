import numpy as np

import matplotlib.pyplot as plt



def symlog(x):

    return np.sign(x) * np.log(1 + np.abs(x))



x = np.linspace(-3, 3, 400)

y = symlog(x)



plt.figure()

plt.plot(x, y)

plt.xlabel("x")

plt.ylabel("symlog(x)")

plt.title("Symlog Function (-3 to 3)")

plt.show()
