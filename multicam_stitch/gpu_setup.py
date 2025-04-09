import cv2
import platform
import subprocess
import os
from .logger_manager import get_logger

logger = get_logger()

# Optional imports for enhanced GPU support
try:
    import pycuda.driver as cuda_driver     # type: ignore
    import pycuda.autoinit                  # type: ignore
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False

try:
    import cupy as cp                       # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import pyopencl as cl                   # type: ignore
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False

GPU_CONFIG = {
    'preferred_backend': os.environ.get('GPU_BACKEND', 'auto'),  # 'cuda', 'opencl', 'cpu', or 'auto'
    'fallback_to_cpu': True,
    'memory_strategy': 'default',  # 'default', 'pinned', 'batch'
    'selected_device_index': 0,
    'backend_initialized': False,
    'devices': []
}

def _enumerate_devices():
    """Enumerate available CUDA and OpenCL devices"""
    devices = []
    # CUDA devices
    if PYCUDA_AVAILABLE:
        try:
            cuda_driver.init()
            for i in range(cuda_driver.Device.count()):
                dev = cuda_driver.Device(i)
                devices.append({
                    'index': i,
                    'name': dev.name(),
                    'backend': 'cuda',
                    'total_mem': dev.total_memory(),
                    'compute_capability': dev.compute_capability(),
                })
        except Exception as e:
            logger.warning(f"PyCUDA device enumeration failed: {e}")

    # CuPy devices (redundant if PyCUDA present, but safe)
    if CUPY_AVAILABLE and not devices:
        try:
            n_devices = cp.cuda.runtime.getDeviceCount()
            for i in range(n_devices):
                with cp.cuda.Device(i):
                    attrs = cp.cuda.runtime.getDeviceProperties(i)
                    devices.append({
                        'index': i,
                        'name': attrs['name'].decode() if isinstance(attrs['name'], bytes) else attrs['name'],
                        'backend': 'cuda',
                        'total_mem': attrs['totalGlobalMem'],
                        'compute_capability': (attrs['major'], attrs['minor'])
                    })
        except Exception as e:
            logger.warning(f"CuPy device enumeration failed: {e}")

    # OpenCL devices
    if PYOPENCL_AVAILABLE:
        try:
            for platform_ in cl.get_platforms():
                for device in platform_.get_devices():
                    devices.append({
                        'index': 0,  # OpenCL device index within platform not tracked here
                        'name': device.name,
                        'backend': 'opencl',
                        'platform': platform_.name,
                        'max_mem_alloc': device.max_mem_alloc_size,
                        'global_mem_size': device.global_mem_size,
                        'max_compute_units': device.max_compute_units
                    })
        except Exception as e:
            logger.warning(f"PyOpenCL device enumeration failed: {e}")

    return devices

def _init_gpu_backend():
    """Initialize the best or configured GPU backend"""
    gpu_info = {
        'backend': None,
        'device_name': None,
        'platform': platform.system()
    }

    devices = _enumerate_devices()
    GPU_CONFIG['devices'] = devices

    preferred = GPU_CONFIG.get('preferred_backend', 'auto').lower()

    try:
        # Explicit backend selection
        if preferred == 'cuda':
            if any(d['backend'] == 'cuda' for d in devices):
                cv2.setUseOptimized(True)
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    cv2.cuda.setDevice(GPU_CONFIG.get('selected_device_index', 0))
                    gpu_info['backend'] = 'cuda'
                    gpu_info['device_name'] = cv2.cuda.getDevice()
            else:
                logger.warning("Preferred CUDA backend not available")
        elif preferred == 'opencl':
            if any(d['backend'] == 'opencl' for d in devices):
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    device = cv2.ocl.Device.getDefault()
                    gpu_info['backend'] = 'opencl'
                    gpu_info['device_name'] = device.name() if device else 'Unknown OpenCL device'
            else:
                logger.warning("Preferred OpenCL backend not available")
        elif preferred == 'cpu':
            gpu_info['backend'] = None
            gpu_info['device_name'] = None
        elif preferred == 'auto':
            # Auto-detect: CUDA preferred, then OpenCL, then Metal
            if (cv2.__version__.find('cuda') > 0 or
                (hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0)):
                cv2.cuda.setDevice(0)
                gpu_info['backend'] = 'cuda'
                gpu_info['device_name'] = cv2.cuda.getDevice()
            elif cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    device = cv2.ocl.Device.getDefault()
                    gpu_info['backend'] = 'opencl'
                    gpu_info['device_name'] = device.name() if device else 'Unknown OpenCL device'
            else:
                gpu_info['backend'] = None
                gpu_info['device_name'] = None
        else:
            logger.warning(f"Unknown preferred backend: {preferred}")
            gpu_info['backend'] = None
            gpu_info['device_name'] = None

        cv2.setUseOptimized(True)

        if gpu_info['backend']:
            logger.info(f"GPU acceleration enabled: {gpu_info['backend']} on {gpu_info['device_name']}")
        else:
            logger.warning("No GPU acceleration available, falling back to CPU")

        GPU_CONFIG['backend_initialized'] = True
        return gpu_info

    except Exception as e:
        logger.warning(f"Failed to initialize GPU acceleration: {e}")
        return {'backend': None, 'device_name': None, 'platform': platform.system()}

GPU_INFO = _init_gpu_backend()

def get_gpu_device():
    """Get current GPU backend"""
    return GPU_INFO.get('backend', None)

def get_available_devices():
    """Return list of detected GPU devices"""
    return GPU_CONFIG.get('devices', [])

def select_backend(backend_name):
    """Select GPU backend explicitly"""
    GPU_CONFIG['preferred_backend'] = backend_name
    global GPU_INFO
    GPU_INFO = _init_gpu_backend()
    return GPU_INFO

def create_gpu_mat(frame, pinned=False):
    """
    Move frame to GPU memory using selected backend.
    pinned: if True and supported, use pinned memory (future use)
    """
    try:
        backend = GPU_INFO.get('backend')
        if backend == 'cuda':
            # TODO: Use pinned memory with CuPy/PyCUDA if pinned=True
            return cv2.cuda.GpuMat(frame)
        elif backend in ('opencl', 'metal'):
            return cv2.UMat(frame)
        else:
            return frame
    except Exception:
        return frame
