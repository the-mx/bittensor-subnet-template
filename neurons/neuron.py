from abc import ABC

from template import __spec_version__ as spec_version


class BaseNeuron(ABC):
    spec_version: int = spec_version
