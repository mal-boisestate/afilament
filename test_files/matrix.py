import numpy as np
import time




A = np.random.random((2000, 1000))
B = np.random.random((1000, 500))
start = time.time()
C = new_matrix = np.matmul(A, B)
end = time.time()
print(f"Time is {end - start}")
np.savetxt('matrix_big_1.txt',A,fmt='%.2f')
np.savetxt('matrix_big_2.txt',B,fmt='%.2f')
np.savetxt('matrix_big_result.txt',C,fmt='%.2f')