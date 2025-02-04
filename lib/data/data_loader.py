import importlib
from data import DataLoader, DataloderX


def find_dataset_using_name(dataset_name):
    # Given the option --dataset_mode [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name.lower() + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls
            break

    if dataset is None:
        print("In %s.py, there should be a class with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
        exit(0)

    return dataset


def customer_data_loader(cfg, phase, **kargs):
    dataset_class = find_dataset_using_name(cfg.DATASET.NAME)
    dataset_obj = dataset_class(cfg, phase, **kargs)

    batchSize = cfg.TRAIN.BATCH_SIZE if phase=='train' else cfg.TEST.BATCH_SIZE
    shuffle = cfg.TRAIN.SHUFFLE if phase=='train' else False
    drop_last = cfg.TRAIN.DROP_LAST

    if cfg.TRAIN.PRE_FRFETCH:
        data_loader = DataloderX(dataset=dataset_obj,
                                 batch_size=batchSize,
                                 shuffle=shuffle,
                                 num_workers=int(cfg.WORKERS),
                                 pin_memory=True,
                                 drop_last=drop_last)
    else:
        data_loader = DataLoader(dataset=dataset_obj,
                                 batch_size=batchSize,
                                 shuffle=shuffle,
                                 num_workers=int(cfg.WORKERS),
                                 pin_memory=True)

    return data_loader
