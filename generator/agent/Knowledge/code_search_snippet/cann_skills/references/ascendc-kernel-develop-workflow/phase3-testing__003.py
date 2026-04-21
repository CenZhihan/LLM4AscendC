import numpy as np

output = np.fromfile('output.bin', dtype=np.float32)
golden = np.fromfile('golden.bin', dtype=np.float32)

is_close = np.allclose(output, golden, rtol=rtol, atol=atol)
