import numpy as np
x = np.random.rand(12, 5000).astype(np.float32)
np.save("sample.npy", x)