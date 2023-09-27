Reimplemtation of Karpathy's micrograd in
1. Numpy  : Implements a Tensor backed by numpy array with grads for matmul and mean ops 
2. Torch  : Uses torch's tensor and grad. Adds layer and mlp on top
3. Triton : Use triton's matmul and add implementation. Inference only.

This was done as an exercise to learn torch/triton.
