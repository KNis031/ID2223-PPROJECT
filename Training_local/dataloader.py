import numpy as np
import h5py
from torch.utils import data


class MelSpecDataset(data.Dataset):
    def __init__(self, dir=''):
        self.dir = dir
        self.f = h5py.File(dir, 'r')
        self.dataset = self.f['dataset']

    def __getitem__(self, idx):
        point = self.dataset['data'][idx, :, :]
        label = self.dataset['label'][idx, :]

        return np.array(point).astype('float32'), np.array(label).astype('float32')

    def __len__(self):
        return self.dataset['data'].shape[0]


def split_ids(dir='', split_r=0.3, shuffle=True):
    f = h5py.File(dir, 'r')
    dataset = f['dataset']
    N, B, T = dataset['data'].shape
    val_n = int(N*split_r)
    if shuffle:
        val_split = np.random.randint(0, N, val_n)
        mask = np.ones(N, dtype=bool)
        mask[val_split] = False
        tr_split = np.arange(0, N)[mask]
    else:
        val_split = np.arange(0, val_n)
        tr_split = np.arange(val_n, N)

    f.close()
    return tr_split, val_split


def main():
    import matplotlib.pyplot as plt
    split_r = 0.3
    split_shuffle = True
    load_shuffle = True
    batch_size = 128
    train_data_path = '/Users/karlsimu/Programming/Datasets/spec_tags_top_1000'
    test_data_path = '/Users/karlsimu/Programming/Datasets/spec_tags_top_1000_val'

    tr_split, val_split = split_ids(dir=train_data_path, split_r=split_r,
                                    shuffle=split_shuffle)

    full_set = MelSpecDataset(dir=train_data_path)
    train_set = data.Subset(full_set, tr_split)
    validation_set = data.Subset(full_set, val_split)
    train_loader = data.DataLoader(dataset=train_set, shuffle=load_shuffle, batch_size=batch_size)
    validation_loader = data.DataLoader(dataset=validation_set, shuffle=load_shuffle, batch_size=batch_size)

    test_set = MelSpecDataset(dir=test_data_path)
    test_loader = data.DataLoader(dataset=test_set, shuffle=load_shuffle, batch_size=batch_size)

    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="viridis")
    plt.show()
    print(f"Label: {label}")

    # i = 0
    # for feat, labl in iter(train_loader):
    #     i += 1
    #     print(i)


if __name__ == "__main__":
    main()
