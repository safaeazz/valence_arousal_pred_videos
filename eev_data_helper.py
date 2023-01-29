from utils import *

def chunk_pad(it, size, padval=None):
    it = chain(iter(it), repeat(padval))
    return np.array(iter(lambda: tuple(islice(it, size)), (padval,) * size))


def twolists(l1, l2):
    long_list = [sum(n) for n in zip_longest(l1, l2, fillvalue = 0)]
    #print(long_list)
    #long_list = [[x1],[x2]]
    return long_list

def check_len(data, labels):
    n_data, n_lab, sample_len = [],[],[]
    for i in range(len(data)):
        d = data[i]
        l = labels[i]
        min_len = min([len(d),len(l)])
        n_data.append(d[0:min_len])
        n_lab.append(l[0:min_len])
        sample_len.append(min_len)
    return n_data, n_lab, sample_len


def pad_data_max(videos,labels,max_len):

    A = np.zeros([1,224,224,3],dtype=np.float) 
    L = np.zeros([1,15],dtype=np.float) 
    data = []
    lab = []
    for i in range(len(videos)):
        v = videos[i]
        l = labels[i]
        if len(v) == max_len:
            continue
        else:
            n = (max_len - len(v))
            A1 = np.repeat(A, repeats=n,axis =0)
            L1 = np.repeat(L, repeats=n,axis =0)
            v_n = np.concatenate((v,A1),axis =0)
            l_n = np.concatenate((l,L1),axis =0)
            data.append(v_n)
            lab.append(l_n)
                
    return data,lab

def interpolate_to_min(videos,labels,min_len):
    data = []
    lab = []
    for i in range(len(videos)):
        v = videos[i]
        l = labels[i]
        #if len(v) != min_len:
        v_n = v[:min_len]
        l_n = l[:min_len]
        
        data.append(v_n)
        lab.append(l_n)

    return data, lab

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def resize_pad(img, target=224):
    h, w = img.shape[:2]
    if h < w:
        pad_h = w - h
        pad_w = 0
    else:
        pad_w = h - w
        pad_h = 0

    img = np.pad(img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                 mode='constant')
    if img.shape[0] != img.shape[1] or img.shape[2] != 3:
        print('Error in here, please stop ', img.shape)
        sys.exit(0)
    img = cv2.resize(img, (target, target))
    return img

def read_video2(path, video_info,num_segments,img_size,max_frames):
    
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        return []
    else:
        
        duration = total_frames/fps
        n_clips = round(duration)
    
        print('------------------------------------------------------------------')
        print( '|fps|', fps,'|total frames|',total_frames,'|duration|',round(duration,2),'sec',(duration%3600)//60,'mins')
        
        vd_frames = []
        selected_frames = []
        frame_count = 1
    
        while True:
            
            ret, frame = cap.read()
            if not ret:
                break
        
            frame = cv2.cvtColor(cv2.resize(frame, (img_size,img_size)), cv2.COLOR_BGR2RGB)
    
            vd_frames.append(frame)
    
            if frame_count == round(fps/6) :
                #use_indexes = np.linspace(1, round(fps/6), num=max_frames, dtype=int, endpoint=False)
                #vd_frames1 = [vd_frames[x] for x in use_indexes]            
                #clips.append(vd_frames1)
                if round(fps/6) >2:
                    random_idx = random.choice(range(round(fps/6)-1)) # 1frame/seg(1/6 sec)
                else:
                    random_idx = random.choice(range(round(fps/6)))
                #cv2.imshow('selected idx',vd_frames[random_idx])
                #cv2.waitKey(500)
                selected_frames.append(vd_frames[random_idx])
                vd_frames[:] = []
                frame_count = 0
                #del vd_frames1
                #gc.collect()
            frame_count += 1
    cap.release()
    
    return selected_frames#clips#,vd_frames



def read_video(path,video_info,train_df):
    print('current video ', path)
    #video = cv2.VideoCapture(path+video_info[0]+'.mp4')
    video = cv2.VideoCapture(path)

    
    vid_label = np.array(train_df.iloc[video_info[3]:video_info[2],2:17])
    #print((train_df.iloc[vid[3]:vid[2],2:]).head(5))
    timestamps = np.array(train_df.iloc[video_info[3]:video_info[2],1])
    diff = np.diff(timestamps)
    #timestamps_sec = [t/1000 for t in timestamps]
    total_duration = video_info[1]/1000
    n_timestamps =  vid_label.shape[0]
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))#in the paper, the frames were read at 1hz/6hz?
    fps = int(video.get(cv2.CAP_PROP_FPS))
    #n_frame_per_seg = round(total_frames/n_timestamps)
    n_frame_per_seg = round((1/6)*diff[0])
    labels_per_frame = []


    #print('------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    #print('|ID|',vid[0],'fps', fps,'|total frames|',total_frames,'|total timestamps|', timestamps.shape,vid_label.shape,'|video_len|',total_duration,'|frames per seg|',n_frame_per_seg)
    #print('------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    vd_frames = []
    frame_count = 0
    multiplier = diff[0]
 
    while True:

        ret, frame = video.read()
        frameId =  int(round(video.get(cv2.CAP_PROP_POS_MSEC)))

        if not ret:
            break
        frame = cv2.cvtColor(cv2.resize(frame,(224,224)), cv2.COLOR_BGR2RGB)
        vd_frames.append(frame)

    vd_frames = np.array(vd_frames)
    #labels_per_frame = np.repeat(vid_label,n_frame_per_seg)

    labels_per_frame = np.repeat(vid_label,n_frame_per_seg,axis=0)
    #video.release()
    min_len = np.min([vd_frames.shape[0],labels_per_frame.shape[0]])
    vd_frames = vd_frames[0:min_len]
    labels_per_frame = labels_per_frame[0:min_len]
    return vd_frames,labels_per_frame

def flatten(t):
    return [item for sublist in t for item in sublist]

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_feat_df(df):

    data_id = df.groupby(["YouTube ID"])
    video_len_df = (data_id['YouTube ID','Timestamp (milliseconds)','idx'].agg(['last']).stack())
    video_st_idx = (data_id['YouTube ID','idx'].agg(['first']).stack())    
    video_len_df_arr = video_len_df.to_numpy()
    video_st_idx_arr = video_st_idx.to_numpy()
    orderedList =  np.array(sorted(list(video_st_idx_arr), key=lambda x: list(video_len_df_arr[:,0])))
    last_col = orderedList[:,-1]
    video_feat = np.column_stack((video_len_df_arr,last_col)) #[video_id,lenght(last time stamp value),last_idx,start_idx]
    #print('video feat',len(video_feat))
    return video_feat
    

def get_videos_labels(path,videos_id,video_feat,df,img_size,max_frames):   

    videos = []
    labels = []
    len_data = []
    i = 0
    for vid in video_feat[0:3]:
 
        if vid[0] in videos_id: # if vid_id in directory

            vid_label = np.array(df.iloc[vid[3]:vid[2],2:17])
            vd_frames = read_video2(path+vid[0]+'.mp4',vid, len(vid_label),img_size,max_frames)
           #vd_frames = flatten(vd_clips)
                        
            ## divide into clips 
            #print('num frames ',len(vd_frames))
            #n = 60 * 6 # clips of 60 seconds
            #vd_clips = chunk_pad(vd_frames, n, padval=None)
            #vd_labels_per_clips = chunk_pad(vid_label, n, padval=None)
            ##vd_clips = np.dstack([ np.array(vd_frames[i:i + n]) for i in range(0, len(vd_frames), n)])
            ##vd_labels_per_clips = np.dstack([ np.array(vid_label[i:i + n]) for i in range(0, len(vid_label), n)])
            #print('vid labels',vd_clips.shape,vd_labels_per_clips.shape)
            if len(vd_frames) != 0:

                print(i, ' len==', len(vd_frames))
     
                videos.append(vd_frames)
                labels.append(vid_label)
                #len_data.append(len(vd_clips))
                del vd_frames
                gc.collect()
                i = i +1

    videos ,labels, videos_len = check_len(videos, labels)   
    return videos, labels, videos_len 


def read_eev_data(path,csv_train_path,csv_valid_path,img_size,max_frames):

    videos = os.listdir(path)
    videos_id = [vid.split(".")[0] for vid in videos]
    
    train_df = pd.read_csv(csv_train_path)
    valid_df = pd.read_csv(csv_valid_path)

    train_df['idx'] = pd.Series(train_df.index)
    valid_df['idx'] = pd.Series(valid_df.index)

    video_train_feat = get_feat_df(train_df)
    video_valid_feat = get_feat_df(valid_df)

    train_videos, train_labels, train_vid_len = get_videos_labels(path,videos_id,video_train_feat,train_df,img_size,max_frames) 
    valid_videos, valid_labels, valid_vid_len = get_videos_labels(path,videos_id,video_valid_feat,valid_df,img_size,max_frames) 

    return train_videos, train_labels, valid_videos, valid_labels, train_vid_len, valid_vid_len
