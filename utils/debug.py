from typing import Any
import logging

def info(msg:Any):
    print(msg) or logging.getLogger().info(msg) # type: ignore