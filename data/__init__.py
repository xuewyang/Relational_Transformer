from .field import RawField, Merge, ImageDetectionsField, BBoxField, RelPairField, RelLabelField, TextField
from .dataset import COCO
from .dataset_r import COCOR
from torch.utils.data import DataLoader as TorchDataLoader


class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)
