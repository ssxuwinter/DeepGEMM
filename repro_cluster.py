import os
import sys
import shutil
import signal
import torch

cache_dir = os.path.expanduser("~/.deep_gemm/cache")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

import deep_gemm
from deep_gemm.testing import calc_diff

sys.path.insert(0, 'tests')
from generators import generate_normal, KernelType

# Use a non-trivial shape that should trigger cluster_m=2 multicast
# Normal GEMM with m=4096, nk picked to get large enough num_blocks on SM count
m, n, k = 4096, 4096, 7168

print(f"Testing cluster multicast on m={m}, n={n}, k={k}")
from deep_gemm.utils.math import align
from deep_gemm.testing.numeric import calc_diff

# Reuse generator for FP8 NT layout (K-major A + K-major B)
from generators import MajorTypeAB, QuantConfig
quant_config = QuantConfig.get_list_from_dtype(torch.float8_e4m3fn)[0]
kernel_type = KernelType.Kernel1D2D
use_ue8m0 = False

a, b, c, d, ref_d = generate_normal(
    m, n, k,
    MajorTypeAB.KMajor, MajorTypeAB.KMajor,
    False, torch.bfloat16, kernel_type,
    use_ue8m0=use_ue8m0,
    quant_config=quant_config,
)

print("Running fp8_fp4_gemm_nt (will hang if bug still present)...")

def handler(signum, frame):
    print("!! HANG DETECTED (timeout) !!")
    sys.exit(1)

signal.signal(signal.SIGALRM, handler)
signal.alarm(30)
deep_gemm.fp8_fp4_gemm_nt(
    a, b, d, c=c,
    disable_ue8m0_cast=not use_ue8m0,
    recipe=quant_config.get_recipes()[0],
    recipe_a=quant_config.get_recipes()[1],
    recipe_b=quant_config.get_recipes()[2],
)
torch.cuda.synchronize()
signal.alarm(0)

diff = calc_diff(d, ref_d)
print(f"Diff = {diff:.5f}; threshold = {quant_config.max_diff()}")
assert diff < quant_config.max_diff(), "numerical mismatch!"
print("PASSED")
