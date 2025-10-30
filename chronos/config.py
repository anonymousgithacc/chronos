from enum import Enum
from defaults import GPU_SPECS, MODEL_SPECS, DATASET_SPECS


class Models(Enum):
    Mixer = "Mixer"
    Resmlp = "Resmlp"
    Asmlp = "Asmlp"
    
    def __str__(self):
        return self.value
    
    @property
    def specs(self):
        return MODEL_SPECS.get(self.value, {})


class Datasets(Enum):
    Cifar100 = "Cifar100"
    TinyImageNet = "TinyImageNet"
    STL10 = "STL10"

    def __str__(self):
        return self.value
    
    @property
    def specs(self):
        return DATASET_SPECS.get(self.value, {})


class Optimizers(Enum):
    Adam = "adam"
    SGD = "sgd"
    RMSProp = "rmsprop"

    def __str__(self):
        return self.value


class Devices(Enum):
    A100 = "A100"
    V100 = "V100"
    T4 = "T4"
    H100 = "H100"
    A40 = "A40"
    RTX5090 = "RTX5090"
    L4 = "L4"
    L40S = "L40S"
    P4 = "P4"
    P100 = "P100"

    def __str__(self):
        return self.value
    
    @property
    def specs(self):
        return GPU_SPECS.get(self.value, {})