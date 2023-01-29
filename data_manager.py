from utils import *


def interpolate_to_max(l,data, labels1, labels2):
    
    tr_data = [v[:l] for v in data ]
    
    vv_labels = []
    aa_labels = []

    for v in labels1:
        vv_labels.append(v[:l])
    for a in labels2:
        aa_labels.append(a[:l])
    labels = [vv_labels,aa_labels]
    return tr_data, vv_labels, aa_labels


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunks_var(data, sizes):
    it = iter(data)
    return  [[next(it) for _ in range(size)] for size in sizes]
        
def to_tensor(data,device): 
    #return torch.stack([torch.Tensor(i) for i in data]).to(device)
    return torch.from_numpy(data).float().to(device)

def len_lst_to_idx(lst):
    idx = []
    st = 0
    end = 0
    for l in lst:
        end = (end + l)
        idx.append((st,end-1))
        st =( st + l)-1
    return idx

def imerge(a, b):
    for i, j in zip(a,b):
        yield i
        yield j

def save_to_bin(filename, data):
    # data shape: (num_samples,num_frames,h,w,ch) for liris
    # data shape: (num_samples,h,w,ch) for eev

    shape = data.shape
    num_samples = shape[0]
    rows = shape[1]
    #cols = shape[1]*shape[2]*shape[3] # liris
    cols = shape[2]*shape[3]*shape[4] # eev

    # 5d to 3d
    #data = data.reshape(num_samples,cols) #liris
    data = data.reshape(num_samples,rows,cols)
    dtype = np.float32
    #
    with open(filename, 'wb') as fout:
        # write a header that contains the total number of samples and the rows and columns per sample
        #fout.write(np.array((num_samples, cols), dtype=np.int32).tobytes())# liris
        fout.write(np.array((num_samples, rows, cols), dtype=np.int32).tobytes()) # eev

        for i in tqdm(range(num_samples)):
            sample = np.array(data[i]).astype(dtype)
            fout.write(sample.tobytes())

def save_to_txt(l,file):  
    with open(file, 'w') as f:
        for item in l:
            f.write("%s\n" % item)
    


def binary_reader1(filename, start=None, end=None, dtype=np.float32):
    itemsize = np.dtype(dtype).itemsize
    with open(filename, 'rb') as fin:
        num_samples, rows = np.frombuffer(fin.read(2 * np.dtype(np.int32).itemsize), dtype=np.int32)

        start = start if start is not None else 0
        end = end if end is not None else num_samples
        blocksize = itemsize * rows #* cols
        start_offset = start * blocksize
        #print(start,end,itemsize,blocksize)
        fin.seek(start_offset, 1)
        for _ in range(start, end):
            yield process_data(np.frombuffer(fin.read(blocksize), dtype=dtype))#.reshape(rows, cols).copy()

def iter_from_file(file,start,end):
    #arr = np.loadtxt(file)
    #return [ np.array(vid_label[i:i + n]) for i in range(0, len(vid_label), n)]
    with open(file) as f:
        yield f.readlines()[start:end]
        #for _ in range(start,end):
        #yield iter(arr[start:end,:])
        
def binary_reader2(filename, start=None, end=None, dtype=np.float32):
    itemsize = np.dtype(dtype).itemsize
    #lines = np.loadtxt(file1).astype(float)
    with open(filename, 'rb') as fin:
        num_samples, rows, cols = np.frombuffer(fin.read(3 * np.dtype(np.int32).itemsize), dtype=np.int32)
        start = start if start is not None else 0
        end = end if end is not None else num_samples
        blocksize = itemsize * rows * cols
        start_offset = start * blocksize
        fin.seek(start_offset, 1)
        for _ in tqdm(range(start, end)):
            yield process_data(np.frombuffer(fin.read(blocksize), dtype=dtype).reshape(rows, 224,224,3).copy())
            #yield np.frombuffer(fin.read(blocksize), dtype=dtype).reshape(rows,cols).copy()


class BinaryIterableDataset(IterableDataset):
    def __init__(self, filename, file1,start=None, end=None,dtype=np.float32):
        super().__init__()
        self.filename = filename
        self.start = start
        self.end = end
        self.dtype = dtype
        self.file1 = file1

    def __iter__(self):
        #arr = np.array(list(binary_reader2(self.filename, self.start, self.end, self.dtype)))
        #arr = np.array(list(binary_reader2(bin_data_file))) #liris
        iter1 = binary_reader2(self.filename,self.start, self.end, self.dtype)
        iter2 = iter_from_file(self.file1,self.start,self.end)
        return iter1#zip(iter1,iter2)


def read_files(bin_data_file, txt_labels_file, txt_len_file, img_size,channel):

    #arr = np.array(list(binary_reader1(bin_data_file))) # eev
    #arr = arr.reshape(arr.shape[0],img_size,img_size,channel)

    arr = np.array(list(binary_reader2(bin_data_file))) #liris
    arr = arr.reshape(arr.shape[0],arr.shape[1],img_size,img_size,channel)
    
    data = []
    for clip in arr:
        clip_imgs = [np.array(Image.fromarray(img.astype(np.uint8))) for img in clip]
        data.append(clip_imgs)

    labels = np.loadtxt(txt_labels_file)    
    len_lst = np.loadtxt(txt_len_file)    

    return np.array(data), labels, len_lst

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def process_data(arr):
    #arr = arr.reshape(arr.shape[0],arr.shape[1],224,224,3)   
    #data = []
    #for clip in arr:
    #    data.append([np.array(Image.fromarray(img.astype(np.uint8))) for img in clip])
    #print('----',arr.shape)
    #x = (x - mean) / std
    return np.array([normalize(np.array(Image.fromarray(img.astype(np.uint8)))) for img in arr])

def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img

def scale(data):
    mean_dct = np.mean(data)
    std_dct = np.std(data)
    data2 =  (data-mean_dct)/std_dct
    return data2

def collate_fn_2(data):
    samples, _, _,_ = zip(*data)
    lenghts
    max_len = max(lengths)
    n_samples = torch.zeros((len(samples),max_len+1, 3,224,224))
    n_labels = torch.zeros((len(samples),max_len+1, 2))
    for i in range(len(samples)):

        s = samples[i]
        l = torch.Tensor(labels[i])
        s = torch.stack([torchvision.transforms.functional.to_tensor(pic) for pic in s])
        j, k,l,m = s.size(0), s.size(1), s.size(2), s.size(3)

        n_samples[i] = torch.cat([s, torch.zeros((max_len - j+1, k,l,m))])
        n_labels[i] = torch.cat([ torch.Tensor(labels[i]), torch.zeros((max_len - j+1, 2))])
    return n_samples.float(),n_labels.float(), lengths

