import sys
from types import ModuleType, FunctionType
from gc import get_referents
import psutil

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


# Memory Information
def getMemory():
    # get the memory details
    svmem = psutil.virtual_memory()
    print("    Memory:  ", f"Total: {get_size(svmem.total)}", f"  Available: {get_size(svmem.available)}",
          f"  Used: {get_size(svmem.used)}", f"  Percentage: {svmem.percent}%")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    print("    Swap:  ", f"Total: {get_size(swap.total)}", f"  Free: {get_size(swap.free)}",
          f"  Used: {get_size(swap.used)}", f"  Percentage: {swap.percent}%")

if __name__ == '__main__':
    getMemory()
