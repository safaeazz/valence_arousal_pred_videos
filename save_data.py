import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader

# data shape: (num_samples,num_frames,h,w,ch)
filename = 'data.bin'
shape = (np.array(all_movies)).shape
num_samples = shape[0]
rows = shape[1]
cols = shape[2]*shape[3]*shape[4]
# 5d to 3d
data = (np.array(all_movies)).reshape(num_samples,rows,cols)
dtype = np.float32
#
with open(filename, 'wb') as fout:
    # write a header that contains the total number of samples and the rows and columns per sample
    fout.write(np.array((num_samples, rows, cols), dtype=np.int32).tobytes())
    for i in tqdm(range(num_samples)):
        # random placeholder
        sample = np.random.randn(rows, cols).astype(dtype)
        # write data to file
        fout.write(sample.tobytes())

with open('your_file.txt', 'w') as f:
    for item in my_list:
        f.write("%s\n" % item)

print('file saved')


def binary_reader(filename, start=None, end=None, dtype=np.float32):
    itemsize = np.dtype(dtype).itemsize
    with open(filename, 'rb') as fin:
        num_samples, rows, cols = np.frombuffer(fin.read(3 * np.dtype(np.int32).itemsize), dtype=np.int32)
        start = start if start is not None else 0
        end = end if end is not None else num_samples
        blocksize = itemsize * rows * cols
        start_offset = start * blocksize
        fin.seek(start_offset, 1)
        for _ in range(start, end):
            yield np.frombuffer(fin.read(blocksize), dtype=dtype).reshape(rows, cols).copy()



class BinaryIterableDataset(IterableDataset):
    def __init__(self, filename, start=None, end=None, dtype=np.float32):
        super().__init__()
        self.filename = filename
        self.start = start
        self.end = end
        self.dtype = dtype

    def __iter__(self):
        return binary_reader(self.filename, self.start, self.end, self.dtype)


dataset = BinaryIterableDataset('data.bin')
for sample in tqdm(dataset):
    pass


gen = binary_reader(filename)
arr = np.array(list(gen))
print('arr shape' , arr.shape)
print('original shape ', (arr.reshape(num_samples,shape[1],shape[2],shape[3],shape[4])).shape)


