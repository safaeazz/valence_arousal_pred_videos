import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F

from timesformer.models.helpers import load_pretrained
from itertools import repeat

from utils import *
from args import *
from vit import *
from configs import *
from pytorch_pretrained_vit import ViT as Previt
from video_dataset import *
#from stam_pytorch import STAM

# the config used for vit (see configs.py)
config = CONFIGS["ViT-B_16"]   
    
# setting args and seed 
args = parser.parse_args()
seed = args.seed
results_path = './results/'

# save config to file
if os.path.isdir(results_path+args.experiment_name) == False:
    os.mkdir(results_path+args.experiment_name)
    plots_folder = os.mkdir(results_path+'/'+args.experiment_name+'/plots')
    logs_folder = os.mkdir(results_path+'/'+args.experiment_name+'/logs')
with open(results_path+args.experiment_name+'/config.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    f.write(str(config.keys()))
    f.write(str(config.values()))


##### setting deterministic setting for torch
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# define gpu/cpu usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)
# start
start = time.time()

# define data path (this is the path used in the server edge-XLIM)

train_path = '/media/DATA/admin-edge/Documents/safaa-project/data/frames/'
test_path = '/media/DATA/admin-edge/Documents/safaa-project/data/test-frames/'

path_audio = '/media/DATA/admin-edge/Documents/safaa-project/data/audio/'
te_path_audio = '/media/DATA/admin-edge/Documents/safaa-project/data/test-audio/'

train_lab = '/media/DATA/admin-edge/Documents/safaa-project/data/labels.txt'
test_lab = '/media/DATA/admin-edge/Documents/safaa-project/data/test_labels.txt'

train_lab ="/media/DATA/admin-edge/Documents/safaa-project/data/media-eval18/train-annotations/"
train_audio= "/media/DATA/admin-edge/Documents/safaa-project/data/media-eval18/train-spectrogram/"
train_frames = "/media/DATA/admin-edge/Documents/safaa-project/data/media-eval18/train-frames/"

test_lab = "/media/DATA/admin-edge/Documents/safaa-project/data/media-eval18/test-annotations/"
test_audio= "/media/DATA/admin-edge/Documents/safaa-project/data/media-eval18/test-spectrogram/"
test_frames = "/media/DATA/admin-edge/Documents/safaa-project/data/media-eval18/test-frames/"


print('[INFO]: reading data...')

# audio config to extract the spectrogram
audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'freqm': 24, 'timem': 192, 'mixup': 0.5, 'skip_norm': True,'mean':-6.421127,'std':4.235824}

root_tr = os.path.join(os.getcwd(), train_path)  
annotation_file_tr = os.path.join(root_tr, 'labels.csv') 

root_te = os.path.join(os.getcwd(), test_path) 
annotation_file_te = os.path.join(root_te, 'test-labels.csv')  


if args.dataset == 'mediaeval18':
    tr_dataset = liris_data_ME18(train_frames, train_audio, train_lab,args.img_size,10)
    train_loader = DataLoader(tr_dataset, collate_fn=collate_fn,batch_size=1)
    
    te_dataset = liris_data_ME18(test_frames, test_audio, test_lab,args.img_size,10)
    test_loader = DataLoader(te_dataset, collate_fn=collate_fn,batch_size=1)

elif args.dataset == 'mediaeval16':
    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(224),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(224),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tr_dataset = VideoFrameDataset(
        root_path=root_tr,
        annotationfile_path=annotation_file_tr,
        num_segments=args.max_frames,
        frames_per_segment=1,
        imagefile_template='{:d}.bmp',
        transform=preprocess,
        test_mode=False
    )
    
    te_dataset = VideoFrameDataset(
        root_path=root_te,
        annotationfile_path=annotation_file_te,
        num_segments=args.max_frames,
        frames_per_segment=1,
        imagefile_template='{:d}.bmp',
        transform=preprocess,
        test_mode=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=args.bsize,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=args.bsize,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )


# finding mean, std for normalization
'''
mean=[]
std=[]
for i, (_,_, labels,_) in enumerate(train_loader):

    #labels = torch.stack(labels)
    #print(labels.shape, labels[0])
    cur_mean = torch.mean(labels)
    cur_std = torch.std(labels)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))

sys.exit()
'''

if args.train_model == 'True':

    # time space transformer model used in this paper: https://arxiv.org/pdf/2102.05095.pdf
    #model1 = TimeSformer(img_size=224, num_classes=2, num_frames=6, attention_type='divided_space_time')  
    #model1 = torchvision.models.video.r3d_18(pretrained = True, progress= True)
    #model1.fc = nn.Linear(512, 2)

    # the model used in this paper: https://arxiv.org/pdf/2103.13915.pdf
    model1 = STAM(
        dim = 512,
        image_size = 224,     # size of image
        patch_size = args.patch,      # patch size
        num_frames = args.max_frames,       # number of image frames, selected out of video
        space_depth = args.depth,     # depth of vision transformer
        space_heads = args.head,      # heads of vision transformer
        space_mlp_dim = 512, # feedforward hidden dimension of vision transformer
        time_depth = args.depth,       # depth of time transformer (in paper, it was shallower, 6)
        time_heads = args.head,       # heads of time transformer
        time_mlp_dim = 512,  # feedforward hidden dimension of time transformer
        num_classes = 2,    # number of output classes
        space_dim_head = 64,  # space transformer head dimension
        time_dim_head = 64,   # time transformer head dimension
        dropout = args.drop,         # dropout
        emb_dropout = 0.0      # embedding dropout
    )
        
    # audio spectrogram transformer: https://arxiv.org/pdf/2104.01778.pdf
    model2 = ASTModel(label_dim=2, \
             fstride=16, tstride=16, \
             input_fdim=128, input_tdim=512, \
                      imagenet_pretrain=True, audioset_pretrain=False)
    
    
    #model1 = VisionTransformer2(config, num_classes=2, zero_head=False, img_size=224, vis=True)

    model1= torch.nn.DataParallel(model1)
    model1 = model1.to(device)
    
    model2= torch.nn.DataParallel(model2)
    model2 = model2.to(device)
    
    # this function is implemented in the following path: models_to_train/train_model.py
    train(train_loader,test_loader,args,model1,model2, device,results_path,'False')

if args.load_model == 'True':


    true, predictions, mse_valence, mse_arousal, ppc_valence, ppc_arousal = infer_from_model(test_loader, device, args,results_path,'True')

    print('mse valence ',mse_valence)
    print('mse arousal ',mse_arousal)
    print('ppc valence ',ppc_valence)
    print('ppc arousal',ppc_arousal)
    
    fig = plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(predictions[:,0], color='b',label='pred. valence')
    plt.plot(true[:,0],color='r',label='true valence')
    plt.legend(fontsize=5)
       
    plt.subplot(2, 1, 2)
    plt.plot(predictions[:,1], color='b',label='pred. arousal')
    plt.plot(true[:,1],color='r',label='true arousal')
    plt.legend(fontsize=5)
    
    plt.suptitle('mse valence= '+str(mse_valence.cpu().detach().numpy())+', mse arousal = '+str(mse_arousal.cpu().detach().numpy())+ '\nppc valence= '+str(ppc_valence)+', ppc arousal = '+str(ppc_arousal),fontsize=10)
    
    plt.savefig(results_path+'/'+args.experiment_name+'/plots/pred-vs-true.pdf',dpi=300)
    plt.close()
    
    #plt.figure(3)
    #sns.heatmap(att_map.cpu().detach().numpy())
    #plt.savefig(results_path+'/'+args.experiment_name+'/plots/att_map.pdf')
    #plt.close()

    param = ['ellapsed time','MSE_valence','MSE_arousal','ppc_valence','ppc_arousal']
    # remove len train, add ppc to table, add ppc library to utils
    res = [(time.time() -start),mse_valence,mse_arousal,ppc_valence,ppc_arousal]
    save_xls(results_path+'/'+args.experiment_name+'/logs/',  param, res)
    
    display_data(X_test1,'original images','attention maps',5,np.array(y_test1),model,device,args.experiment_name)

        






















