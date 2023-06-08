from dataclasses import dataclass

@dataclass
class DatasetConfig:
    pass

@dataclass
class FlowersConfig(DatasetConfig):
    root: str = './.cache/datasets/jpg'
    labels_mat: str = './.cache/datasets/imagelabels.mat'
    splits_mat: str = './.cache/datasets/setid.mat'

@dataclass
class AircraftConfig(DatasetConfig):
    root: str = './.cache/datasets/fgvc-aircraft-2013b'

@dataclass
class DTDConfig(DatasetConfig):
    root: str = './.cache/datasets/dtd'

@dataclass
class BirdConfig(DatasetConfig):
    root: str = './.cache/datasets/birdsnap'

@dataclass
class PetsConfig(DatasetConfig):
    root: str = './.cache/datasets/images'
    root_anno: str = './.cache/datasets/annotations'

@dataclass
class CarsConfig(DatasetConfig):
    root_train: str = './.cache/datasets/cars_train'
    root_test: str = './.cache/datasets/cars_test'
    devkit: str = './.cache/datasets/devkit'
    testlabel: str = './.cache/datasets/cars_test_annos_withlabels.mat'

@dataclass
class Caltech101Config(DatasetConfig):
    root: str = './.cache/datasets/caltech101'

@dataclass
class Caltech256Config(DatasetConfig):
    root: str = './.cache/datasets/caltech256'

@dataclass
class CFPConfig(DatasetConfig):
    root: str = './.cache/datasets/cfp-dataset'

@dataclass
class CIFAR10Config(DatasetConfig):
    root: str = './.cache'

@dataclass
class CIFAR100Config(DatasetConfig):
    root: str = './.cache'

@dataclass
class CIFAR100LTConfig(DatasetConfig):
    root: str = './.cache'
    imb_factor: int = 50


@dataclass
class AllDatasetsConfigs:
    flowers: FlowersConfig = FlowersConfig()
    aircraft: AircraftConfig = AircraftConfig()
    dtd: DTDConfig = DTDConfig()
    birds: BirdConfig = BirdConfig()
    pets37: PetsConfig = PetsConfig()
    pets2: PetsConfig = PetsConfig()
    cars: CarsConfig = CarsConfig()
    caltech101: Caltech101Config = Caltech101Config()
    caltech256: Caltech256Config = Caltech256Config()
    cfp: CFPConfig = CFPConfig()
    cifar10: CIFAR10Config = CIFAR10Config()
    cifar100: CIFAR100Config = CIFAR100Config()
    cifar100lt: CIFAR100Config = CIFAR100LTConfig()
