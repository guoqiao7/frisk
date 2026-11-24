"""Native execution of a dense batch of independent solved-layout SM90 GEMMs."""
import ctypes
import os
import subprocess
from dataclasses import dataclass
from threading import Lock
from weakref import finalize

from gemm_accumulator_slices import compile_accumulator_gemm, read_layouts


@dataclass(frozen=True)
class DenseGemm:
    """FP16 A @ B -> FP32 C, plus the column sum if present in the solved layouts.

    One 256-thread CTA per packed problem. The layout IR supplies the certified
    carriers and writers; k is the dense GEMM extent, not the WGMMA K16 atom.
    """
    layout_ir: str
    k: int

    @classmethod
    def from_open_layout(cls, layout_ir, k, *, resident_atoms, staged_k_tiles=2, frisk_opt=None):
        """Solve a fixed-width native SS GEMM with direct C and optional column sum."""
        if type(resident_atoms) is not int or resident_atoms < 1 or staged_k_tiles not in (1, 2):
            raise ValueError("expected positive resident width and one or two K16 slots")
        solver = frisk_opt or os.environ.get("FRISK_OPT", "frisk-opt")
        option = ("--frisk-protected-store-layout=anchor-solver=integer "
                  f"native-resident-atoms={resident_atoms} native-k-slots={staged_k_tiles}")
        solved = subprocess.run([solver, option, "--canonicalize", "-verify-each"],
                                input=layout_ir, capture_output=True, text=True)
        if solved.returncode:
            raise ValueError(solved.stderr.strip())
        if read_layouts(solved.stdout).slice_contract != (k, resident_atoms, staged_k_tiles):
            raise ValueError("native solve returned a different slice contract")
        return cls(solved.stdout, k)


class NativeGemmImage:
    def __init__(self, binary, info):
        self.asm = {"cubin": binary}
        self._shape = tuple(info["problem_shape"])
        self._column_sum = info["column_sum"]
        self._symbol = info["symbol"].encode()
        self._shared = info["dynamic_shared_bytes"]
        self._kernel = None
        self._lock = Lock()

    @staticmethod
    def _check(status, operation):
        if status:
            raise RuntimeError(f"{operation} failed with CUDA driver status {status}")

    def _load(self):
        # A CUDA library can be used by more than one primary context. Resolve
        # its CUfunction in the current context at each launch.
        with self._lock:
            if self._kernel is not None:
                return
            api = ctypes.CDLL("libcuda.so.1")
            ptr = ctypes.c_void_p
            library, kernel = ptr(), ptr()
            self._check(api.cuLibraryLoadData(ctypes.byref(library), ctypes.c_char_p(self.asm["cubin"]),
                                              None, None, 0, None, None, 0), "cuLibraryLoadData")
            release = finalize(self, api.cuLibraryUnload, library)
            status = api.cuLibraryGetKernel(ctypes.byref(kernel), library, self._symbol)
            if status:
                release()
                self._check(status, "cuLibraryGetKernel")
            self._driver, self._kernel = api, kernel

    def __getitem__(self, grid):
        grid = tuple(grid)
        if len(grid) != 3 or type(grid[0]) is not int or not 1 <= grid[0] < 2**31 or grid[1:] != (1, 1):
            raise ValueError("expected (independent_problems, 1, 1) for native GEMM")

        def launch(a, b, c, r=None):
            import torch
            m, n, k = self._shape
            tensors = [a, b, c]
            shapes = [(grid[0], m, k), (grid[0], k, n), (grid[0], m, n)]
            if self._column_sum != (r is not None):
                raise ValueError("R must match the compiled column-sum contract")
            if r is not None:
                tensors.append(r)
                shapes.append((grid[0], n))
            spans = []
            for i, (tensor, shape) in enumerate(zip(tensors, shapes)):
                dtype = torch.float16 if i < 2 else torch.float32
                if (not tensor.is_cuda or tensor.device != a.device or tensor.dtype != dtype
                        or tuple(tensor.shape) != shape or not tensor.is_contiguous()
                        or tensor.data_ptr() % (16 if i < 2 else 4)):
                    raise ValueError("expected aligned dense batched GEMM tensors on one CUDA device")
                start = tensor.data_ptr()
                end = start + tensor.numel() * tensor.element_size()
                if i >= 2 and any(start < hi and lo < end for lo, hi in spans):
                    raise ValueError("GEMM output overlaps another operand")
                spans.append((start, end))
            with torch.cuda.device(a.device):
                if torch.cuda.get_device_capability(a.device) != (9, 0):
                    raise ValueError("native WGMMA image requires SM90a")
                self._load()
                api = self._driver
                function = ctypes.c_void_p()
                self._check(api.cuKernelGetFunction(ctypes.byref(function), self._kernel), "cuKernelGetFunction")
                shared = self._shared
                self._check(api.cuFuncSetAttribute(function, 8, shared), "cuFuncSetAttribute")
                ptr = ctypes.c_void_p
                pointers = [ptr(t.data_ptr()) for t in tensors]
                if r is None:
                    pointers.append(ptr())
                args = (ptr * 4)(*(ctypes.cast(ctypes.byref(v), ptr) for v in pointers))
                stream = ptr(torch.cuda.current_stream(a.device).cuda_stream)
                self._check(api.cuLaunchKernel(function, *grid, 256, 1, 1, shared, stream, args, None),
                            "cuLaunchKernel")
        return launch


def compile_dense_gemm(problem, **options):
    binary, info = compile_accumulator_gemm(problem.layout_ir, problem.k, **options)
    return NativeGemmImage(binary, info), info
