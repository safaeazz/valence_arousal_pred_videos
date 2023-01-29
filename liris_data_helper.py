import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import *
import torch, torchaudio,torchvision
#def wav2fbank(filename,audio_conf):

def wav2fbank(filename):

    #melbins = audio_conf.get('num_mel_bins')
    melbins = 128
    target_length = 512
    waveform, sr = torchaudio.load(filename)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=melbins, dither=0.0, frame_shift=10)
    
    #target_length = audio_conf.get('target_length')
    n_frames = fbank.shape[0]
    
    p = target_length - n_frames
    
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
        
    return fbank#, mix_lambda
                                                                                                                                                    

def check_len_liris(data, lab1,lab2):
    n_data, n_lab1, n_lab2, len_lst = [],[],[],[]
    for i in range(len(data)):
        d = data[i]
        l1 = lab1[i]
        l2 = lab2[i]
        min_len = min([len(d),len(l1),len(l2)])
        n_data.append(d[0:min_len])
        n_lab1.append(l1[0:min_len])
        n_lab2.append(l2[0:min_len])
        len_lst.append(min_len)
    return n_data, n_lab1,n_lab2,len_lst

def chunk_pad(it, size, padval=None):
    it = chain(iter(it), repeat(padval))
    return np.array(iter(lambda: tuple(islice(it, size)), (padval,) * size))


def chunk_based_on_size1(lst, n):
    A = np.zeros(shape=(224,224,3))
    for x in range(0, len(lst), n):
        each_chunk = lst[x: n+x]

        if len(each_chunk) < n:
            each_chunk = each_chunk + [A for y in range(n-len(each_chunk))]
        yield each_chunk

def chunk_based_on_size2(lst, n):
    for x in range(0, len(lst), n):
        each_chunk = lst[x: n+x]

        if len(each_chunk) < n:
            each_chunk = each_chunk + [0 for y in range(n-len(each_chunk))]
        yield each_chunk

# get num clips
def get_vid_len(videos):
    lens = []
    for vid in videos:
        l = len(vid)
        lens.append(l)
    return lens

# pad to max num clips
#def pad_data_max2(videos,labels_v,labels_a,max_len):
def pad_data_max2(videos,labels,max_len):

    A = np.zeros([1,224,224,3],dtype=np.float) 
    L = np.zeros([1,2],dtype=np.float) 

    data = []
    lab1 = []

    for i in range(len(videos)):
        v = videos[i]
        l1 = labels[i]
        
        if len(v) == max_len:
            data.append(v)
        else:
            n = (max_len - len(v))
            A1 = np.repeat(A, repeats=n,axis =0)
            L1 = np.repeat(L, repeats=n,axis =0)
            v_n = np.concatenate((v,A1),axis =0)
            l_n1 = np.concatenate((l1,L1),axis =0)

            data.append(v_n)
            lab1.append(l_n1)

    return data,lab1

def pad_vec_max(v,max_len):

    A = np.zeros([1,224,224,3],dtype=np.float) 
    #print(np.array(v).shape)
    if len(v) == max_len:
        return v
    else:
        n = (max_len - len(v))
        A1 = np.repeat(A, repeats=n,axis =0)
        v_n = np.concatenate((v,A1),axis =0)            
        return v_n

# check labels consistency
def check_len2(data,lab1,lab2):
  lab11 = []
  lab22 = []
  for i in range(len(data)):
    item = data[i]
    n = len(item)
    l1 = lab1[i]
    l2 = lab2[i]

    if (len(l1) != n or len(l2) != n):
      l1 = l1[:n]
      l2 = l2[:n]
      lab11.append(l1)
      lab22.append(l2)
  return lab11,lab22 
   

# get chunks
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_video_clips(video_path):
    frames = get_video_frames(video_path)
    clips = sliding_window(frames, 16,16)
    return clips


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    return frames


def read_video(path,img_size, max_frames):

    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = total_frames/fps
    n_clips = round(duration)
    threshold = 300000
    #fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    #print('------------------------------------------------------------------')
    #print( '|fps|', fps,'|total frames|',total_frames,'|duration|',round(duration,2),'sec',(duration%3600)//60,'mins')

    frame_count = 1
    clips_count = 0
    clips = []
    vd_frames = []
    i = 0 
    while True:

        ret, frame = video.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(cv2.resize(frame, (img_size,img_size)), cv2.COLOR_BGR2RGB)
        #diff = cv2.subtract(frame, prev_frame)
        #non_zero_count = np.count_nonzero(diff)
        #print('frame_count--',frame_count,non_zero_count)
        #plt.imshow(frame)
        #plt.show()

        #if non_zero_count > threshold:
        #    print("Got P-Frame")
        #prev_frame = frame
        #frame = cv2.cvtColor(cv2.resize(frame,(img_size,img_size)), cv2.COLOR_BGR2RGB)
        vd_frames.append(frame)
        if frame_count == fps :
            
            use_indexes = np.linspace(1, fps, num= max_frames, dtype=int, endpoint=False)
            vd_frames1 = [vd_frames[x] for x in use_indexes]            
            clips.append(vd_frames1)
            vd_frames[:] = []
            frame_count = 0
            del vd_frames1
            clips_count +=1
        frame_count += 1
        i += 1
    #print(len(clips))
    clips = np.squeeze(np.array(clips),axis=1)
    if len(clips)!=18:
        clips =  pad_vec_max(clips,18) 
    return clips


def flatten(t):
    return [item for sublist in t for item in sublist]

def read_liris_data(path_data,path_labels,movies_list,img_size,max_frames):
    directory = os.path.join(path_labels)

    v_labels = []
    a_labels = []
    all_movies = []

    with open(movies_list) as f:
        movies_list = [line.rstrip() for line in f]
    
    step = 0
    
    for name in movies_list:

      file_label = open(path_labels + name + '.txt', 'r')
      lines = file_label.readlines()[1:]
      mean_val = [float(x.split('\t')[1]) for x in lines]
      file_lab = '../../datasets/liris/liris-30/labels/' + name[:-8]
      #if step % 2 == 0:
      #      os.mkdir(file_lab)

      if "Valence" in name:
        #plt.plot(mean_val,color='r',label='valence')
        #plt.legend()
        v_labels.append(mean_val)
    
      elif "Arousal" in name:
        #plt.plot(mean_val,color='b',label='arousal')
        #plt.legend()
        a_labels.append(mean_val)

        #plt.savefig(file_lab + '/label.pdf',dpi=300)
        #plt.close()
        #print('saved')

      if step % 2 == 0:
    
        movie_path = path_data + name[:-8] + '.mp4'
        movie_clips = read_video(movie_path,img_size,max_frames)

        all_movies.append(movie_clips)
        del movie_clips

      step = step + 1

    return all_movies,v_labels,a_labels


def read_liris_media_eval_data(path_data,path_labels,img_size,max_frames):

    v_labels = []
    a_labels = []
    all_movies = []

    movies_ids = [((x.split('_')[1]).split('.')[0]) for x in os.listdir(path_data)  if 'mp4' in x ]
        
    for idx in movies_ids:

      #file_label = [f for f in os.listdir(path_labels) if idx in f ][0]
      file_path = path_labels + 'MEDIAEVAL18_' + idx +'_Valence-Arousal.txt'
      lines = open(file_path, 'r').readlines()[1:]
      valence = [float(x.split()[1]) for x in lines]
      arousal = [float(x.split()[2]) for x in lines]

      movie_path = path_data + 'MEDIAEVAL18_'+ idx +'.mp4'
      print('movieparh ',movie_path)
      movie_clips = read_video(movie_path,img_size,max_frames)

      all_movies.append(movie_clips)
      v_labels.append(valence)
      a_labels.append(arousal)
      del movie_clips

    return all_movies,v_labels,a_labels


def seperate(x):
    average = scipy.ndimage.gaussian_filter(x, 10)
    # this gives me a boolean array with the indices of the upper band.
    idx = x > average
    # return the indices of the upper and lower band
    return idx, ~idx


def read_liris_media_eval_frames(path_imgs, path_audio, path_labels,img_size,s):

    v_labels = []
    a_labels = []
    videos_frames = []
    videos_audio = []

    movies_ids = [((x.split('_')[1]).split('.')[0]) for x in os.listdir(path_imgs)]#  if 'mp4' in x ]

    for idx in movies_ids[0:s]:
        file_path = path_labels + 'MEDIAEVAL18_' + idx +'_Valence-Arousal.txt'
        lines = open(file_path, 'r').readlines()[1:]
        valence = [float(x.split()[1]) for x in lines]
        arousal = [float(x.split()[2]) for x in lines]

        items_1 = os.listdir(path_imgs + 'MEDIAEVAL18_'+ idx)
        items_2 = os.listdir(path_audio + 'MEDIAEVAL18_'+ idx)

        ids1 = sorted([it.split('.')[0] for it in items_1])
        ids2 = sorted([it.split('.')[0] for it in items_2])
        ids = sorted(set(ids1) & set(ids2))
      
        #smile = opensmile.Smile(
        #    feature_set=opensmile.FeatureSet.ComParE_2016,
        #    feature_level=opensmile.FeatureLevel.Functionals,
        #)
        #spect_path = '../../datasets/liris/media-eval/test-spectrogram/MEDIAEVAL18_'+ idx + '/'
        #if(os.path.isdir(spect_path)):
        #  continue 
        #else:
        #  os.mkdir(spect_path)     
        #  audios = []
        frames = []
        for item in ids:  
            
            audio_path = path_audio + 'MEDIAEVAL18_'+ idx + '/' + item + '.png' #'.wav'
            img_path = path_imgs + 'MEDIAEVAL18_'+ idx + '/' + item + '.bmp'
            #print(img_path)
            frame = cv2.imread(audio_path)
            frame = cv2.resize(frame,(img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = scale(frame)
            #frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            #frame = Image.fromarray(frame)
            #frame = scale(frame)
            #print(frame)
            #plt.imshow(frame)
            #plt.show()
            #_, audio = wavfile.read(audio_path)
            #audio, sr = librosa.load(audio_path)
            #print(spect_path)
            '''
            window_size = 1024
            window = np.hanning(window_size)
            stft  = librosa.core.spectrum.stft(audio, n_fft=window_size, hop_length=512, window=window)
            out = 2 * np.abs(stft) / np.sum(window)
            # For plotting headlessly
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
            fig.savefig(spect_path + str(item)+'.png')
  
            #y = smile.process_file(audio_path)
            #y = y.values
            '''
            frames.append(frame)
            #audios.append(audio)
        n=100
        vd_clips = list(chunk_based_on_size1(frames, n))
        v_labels_per_clips = list(chunk_based_on_size2(valence, n))
        a_labels_per_clips = list(chunk_based_on_size2(arousal, n))
        
        #for i in range(len(v_labels_per_clips)):
        #v = v_labels_per_clips[i]
        #a = a_labels_per_clips[i]
        #plt.plot(valence,color='r')
        #id1,_  = seperate(x)
        #print(id1)
        #plt.show()

        print(np.array(vd_clips).shape)
        videos_frames.append(vd_clips)
        del frames
        del vd_clips
        #videos_audio.append(audios)       
        v_labels.append(v_labels_per_clips)
        a_labels.append(a_labels_per_clips)
    return np.concatenate(videos_frames,axis=0),v_labels,a_labels


def read_liris_frames(path_imgs, path_audio, path_labels,movies_list,img_size,s,divide_clips,n,train_model):

    v_labels = []
    a_labels = []
    videos_frames = []
    videos_audio = []

    with open(movies_list) as f:
        movies_list = [line.rstrip() for line in f]

      
    for movie in movies_list:#[0:s]:
        if train_model == 'True':
            a_path = open(path_labels + movie +'_Valence.txt', 'r')
            v_path = open(path_labels + movie +'_Arousal.txt', 'r')
            
            lines_v = a_path.readlines()[1:]        
            lines_a = v_path.readlines()[1:]        
        
            valence = [float(x.split('\t')[1]) for x in lines_v]
            arousal = [float(x.split('\t')[1]) for x in lines_a]
        else:
            print(movie)
            file_path = path_labels + movie +'.txt'
            lines = open(file_path, 'r').readlines()
            
            valence = [float(x.split()[0]) for x in lines]
            arousal = [float(x.split()[1]) for x in lines]
            
        items_1 = os.listdir(path_imgs + movie)
        items_2 = os.listdir(path_audio + movie)

        ids1 = sorted([it.split('.')[0] for it in items_1])
        ids2 = sorted([it.split('.')[0] for it in items_2])
        ids = sorted(set(ids1) & set(ids2))
        frames = []
        
        for item in ids:  
            
            #audio_path = path_audio + 'MEDIAEVAL18_'+ idx + '/' + item + '.png' #'.wav'
            img_path = path_imgs + movie + '/' + item + '.bmp'
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame,(img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            frames.append(frame)
            #audios.append(audio)
        
        if divide_clips == 'True':
            vd_clips = list(chunk_based_on_size1(frames, n))
            v_labels_per_clips = list(chunk_based_on_size2(valence, n))
            a_labels_per_clips = list(chunk_based_on_size2(arousal, n))
            videos_frames.append(vd_clips)
            del frames
            del vd_clips
            #videos_audio.append(audios)       
            v_labels.append(v_labels_per_clips)
            a_labels.append(a_labels_per_clips)
        else:
            videos_frames.append(frames)
            del frames
            v_labels.append(valence)
            a_labels.append(arousal)

    return videos_frames,v_labels,a_labels

def read_liris_discret_data(path_data,path_labels,img_size,max_frames,train,audio_conf):

    v_labels = []
    a_labels = []
    data = []
    #print(path_data)
    if train == 'True':
        
        labels_df = pd.read_csv(path_labels, sep='\t')
        sub_df = pd.DataFrame(labels_df, columns=['name','valenceValue','arousalValue'])
    else:
        sub_df = pd.read_csv(path_labels, sep='\t', names=['name', 'valenceValue','arousalValue'])
    #transform = transforms.Compose([transforms.ToTensor()])
    movies_lst= listdir(path_data)#[f for f in listdir(path_data) if isfile(join(path_data, f))]
    #print(movies_lst)
    d = []
    for movie in movies_lst:

      valence = np.array(sub_df.loc[sub_df['name'] == movie]['valenceValue'])
      arousal = np.array(sub_df.loc[sub_df['name'] == movie]['arousalValue'])
      audio_path = path_data + movie + '/%d.wav'
      #print(valence,arousal)
      #movie_clips = read_video(movie_path,img_size,max_frames)

      #audio = Audio.from_file(audio_path)
      #spectrogram = Spectrogram.from_audio(audio)
      #image = spectrogram.to_image(shape=(224,224)).convert('RGB')
      #audio, sr = torchaudio.load(audio_path)
      #fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sr, use_energy=False,window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
      #print(audio_path,'---->',fbank.shape)
      
      norm_mean = audio_conf.get('mean')
      norm_std = audio_conf.get('std')
      #skip_norm = audio_conf.get('skip_norm') if audio_conf.get('skip_norm') else False
      
      fbank = wav2fbank(audio_path)
      #if not skip_norm:
      fbank = (fbank - norm_mean) / (norm_std * 2)
      #print(type(fbank))
      '''
      d.append(
          {
              'path':str(audio_path)#,
              #'valence':valence[0],
              #'arousal':arousal[0]

          }
      )
      
      df = pd.DataFrame(d)
      df.to_csv('data.csv', index=False,header=False)  
      '''
      #print(df.head())
      #print(fbank.shape)
      #print(np.array(data).shape)
      data.append(fbank)
      #print(np.array(all_movies).shape)
      v_labels.append(valence)
      a_labels.append(arousal)
      #del movie_clips
      #dataframe = pd.read_csv('data.csv')
      #print(dataframe.head())
      #data = (dataframe, False)
      #data = AudioToSpectrogramPreprocessor(dataframe)
      #generate clip df
      #clip_df = make_clip_df(movies_lst)
    
      #create dataset
      #dataset = ClipLoadingSpectrogramPreprocessor(clip_df)
    
      #data =  torch.Tensor(data)
      #print(data.shape)
      #v_labels = np.concatenate(v_labels, axis=0 )
      #a_labels = np.concatenate(a_labels, axis=0 )
    return data,v_labels,a_labels


class liris_data_ME18(torch.utils.data.Dataset):

    def __init__(self, path_imgs, path_audio, path_labels,img_size,s):
        self.path_imgs = path_imgs
        self.path_audio = path_audio
        self.path_labels = path_labels
        self.img_size = img_size
        self.s = s
        self.movies_ids = [((x.split('_')[1]).split('.')[0]) for x in os.listdir(self.path_imgs)]#  if 'mp4' in x ]
        #print(self.movies_ids)
    
    def __len__(self):
        return len(self.movies_ids)

    def __getitem__(self, idx):

        #for idx in movies_ids[0:self.s]:
        #print(idx, self.movies_ids[idx] )
        file_path = self.path_labels + 'MEDIAEVAL18_' + self.movies_ids[idx] +'_Valence-Arousal.txt'
        #print('-file path-',file_path)
        lines = open(file_path, 'r').readlines()[1:]
        #print('len lines - ',len(lines))
        items_1 = os.listdir(self.path_imgs + 'MEDIAEVAL18_'+ self.movies_ids[idx])
        items_2 = os.listdir(self.path_audio + 'MEDIAEVAL18_'+ self.movies_ids[idx])
        
        ids1 = sorted([it.split('.')[0] for it in items_1])
        ids2 = sorted([it.split('.')[0] for it in items_2])
        ids = sorted(set(ids1) & set(ids2))
        
        frames = []
        labels = []

        for index,item in enumerate(ids):  
            #print('item/index',item,index)
            audio_path = self.path_audio + 'MEDIAEVAL18_'+ self.movies_ids[idx] + '/' + item + '.png' #'.wav'
            img_path = self.path_imgs + 'MEDIAEVAL18_'+ self.movies_ids[idx] + '/' + item + '.bmp'
            #print(audio_path,img_path)
            #print('lines index',lines[index])
            label = [float(lines[index].split()[1]),float(lines[index].split()[2])]
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame,(self.img_size, self.img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            labels.append(label)
        #print((np.array(labels)).shape,(np.array(frames)).shape)
        return frames,labels,len(frames)

def collate_fn(data):
    samples, labels, lengths = zip(*data)
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
