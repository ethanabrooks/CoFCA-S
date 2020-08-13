import numpy as np
import h5py
from tqdm import tqdm


class HDF5Store:
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """

    def __init__(
        self,
        datapath,
        dataset="dataset",
        dim=4,
        dtype=np.float32,
        compression="gzip",
        chunk_len=1,
    ):
        self.datapath = datapath
        self.dataset = dataset
        self.dim = dim
        self.i = 0

        with h5py.File(self.datapath, mode="w") as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0, dim),
                maxshape=(None, dim),
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len, dim),
            )

    def append(self, values):
        with h5py.File(self.datapath, mode="a") as h5f:
            dset = h5f[self.dataset]
            i = self.i + len(values)
            dset.resize((i, self.dim))
            dset[self.i : i] = [values]
            self.i = i
            h5f.flush()


# test
if __name__ == "__main__":
    shape = (80, 3)
    hdf5_store = HDF5Store("/tmp/hdf5_store.h5", "X", dim=shape[-1])
    for _ in tqdm(range(int(8e5))):
        hdf5_store.append(np.random.random(shape))
