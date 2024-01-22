from abc import ABC

from template import __spec_version__ as spec_version
from template.utils.misc import ttl_get_block


class BaseNeuron(ABC):
    spec_version: int = spec_version






