import platform, os, pathlib
from tensor import Tensor

OSX = platform.system() == "Darwin"
# download location for datasets, models, etc
_cache_dir = os.path.expanduser("~/Library/Caches" if OSX else "~/.cache") 

def fetch():
    pass

def mnist():
    pass