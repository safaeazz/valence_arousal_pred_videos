from utils import *


def compute_pcc(y_pred, y_true):
    m1 = np.mean(y_pred)
    m2 = np.mean(y_true)
    y_pred_norm = y_pred - m1
    y_true_norm = y_true - m2
    nom = np.sum(y_pred_norm*y_true_norm)
    den = np.sqrt(np.sum(y_pred_norm**2))*np.sqrt(np.sum(y_true_norm**2))

    return nom/den


def to_tensor(data,device): 
    #return torch.stack([torch.Tensor(i) for i in data]).to(device)
    return torch.from_numpy(data).float().to(device)

#def visualize_weights(img,args,device,model):


def save_xls(results_path, row1, row2):
    with open(results_path +'logs.csv', 'w') as f: 
        writer = csv.writer(f) 
        for i in zip(row1, row2):
            writer.writerow(i) 

def save_plots(train_loss, validation_loss, path):
    """ Stores the plots from the trainings in the given path """

    # Show plots
    x = np.arange(len(validation_loss))
    fig = plt.figure(1)
    fig.suptitle('LOSS', fontsize=14, fontweight='bold')

    # LOSS: TRAINING vs VALIDATION
    plt.plot(x, train_loss, '--', linewidth=2, label='train')
    plt.plot(x, validation_loss, label='validation')
    plt.legend(loc='upper right')

    # MIN
    val, idx = min((val, idx) for (idx, val) in enumerate(validation_loss))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val - 0.01),
                 arrowprops=dict(facecolor='black', shrink=0.0005))

    plt.savefig(path+'loss.pdf', dpi=fig.dpi)
    plt.close()



#visualize_SNE(dataset,ids,30,1000,)
def visualize_SNE(X,y,per,iter,n,path):

    N = 10000
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    
    rndperm = np.random.permutation(df.shape[0])
    
    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values
        
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=per, n_iter=iter)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    figure = plt.figure()
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", n),
        data=df_subset,
        #legend="full",
        alpha=0.3
    )
    figure.savefig(path,dpi=300)
    #plt.show()

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def get_att_map(data,model,device):
    
    all_results = []
    for test in data:
        results = []
        #im = torch.tensor(test).permute(2,0,1).float().unsqueeze(0).to(device)
        #_,att_mask = model(im)
        #att_mat = torch.stack(att_mask).squeeze(1)
        #print('-----> test',test.shape)
        clip = torch.tensor(test).permute(3,0,1,2).float().unsqueeze(0).to(device)
        _,att_mask1,att_mask2 = model(clip)

        #print('att_mask',len(att_mask))
        att_mat11 = torch.stack(att_mask1)#.squeeze(1)
        att_mat22 = torch.stack(att_mask2)#.squeeze(1)

        print('stack', att_mat1.shape, att_mat2.squeeze(1).shape)
        
        

        #print('squeeze' ,att_mat1.shape)
        for i in range(4):
            print(i,att_mat1.shape)

            att_mat11 = (att_mat11.squeeze(1))[:,i,:,:,:]#
            att_mat22 = (att_mat22.squeeze(1))[:,:,:,i,i]
            #print('---->',att_mat2[:,:,:,i,:].shape)
            #print('current att mat ',att_mat.shape)
            # Average the attention weights across all heads.
            att_mat1 = torch.mean(att_mat11, dim=1)# heads dim(1 for spatial, and 2 for temporal)
            att_mat2 = torch.mean(att_mat22, dim=2)    
            
            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mat1.size(1))
            aug_att_mat = att_mat1 + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
            #
            ## Recursively multiply the weight matrices
            joint_attentions1 = torch.zeros(aug_att_mat.size())
            joint_attentions1[0] = aug_att_mat[0]            
            for n in range(1, aug_att_mat.size(0)):
                joint_attentions1[n] = torch.matmul(aug_att_mat[n], joint_attentions1[n-1])

            ## Recursively multiply the weight matrices
            joint_attentions2 = torch.zeros(att_mat2.size())
            joint_attentions2[0] = att_mat2[0]            
            for n in range(1, att_mat2.size(0)):
                joint_attentions2[n] = torch.matmul(att_mat2[n], joint_attentions2[n-1])

            v1 = joint_attentions1[-1]
            v2 = joint_attentions2[-1]
            grid_size = int(np.sqrt(att_mat1.size(-1)))
            #print(v.shape,grid_size)
            mask2 = v2.reshape(grid_size, grid_size).detach().numpy()
            mask1 = v1[0, 1:].reshape(grid_size, grid_size).detach().numpy()

            mask1 = mask1 / mask1.max()
            mask1 = cv2.resize(mask1, (224,224))

            mask2 = mask2 / mask2.max()
            mask2 = cv2.resize(mask2, (224,224))

            result1 = show_cam_on_image(test[i,:,:,:], mask1)
            print('result1 ', result1.shape)
            result = show_cam_on_image(result1, mask2)

            #print(result.shape)
            results.append(result)
            #print('len ',len(results))
        all_results.append(results)
    return np.array(all_results)


def display_data(array1, title1,title2,n,Y,model,device,experiment_name):

    indices = np.random.randint(len(array1), size=n)
    print(indices)
    images1 = array1[indices,:]
    Y = Y[indices,:]

    attn = get_att_map(images1,model,device)
    print('attn --',attn.shape,images1.shape)

    plt.figure(figsize=(20, 4))
    
    #for i, (clip1, att, y) in enumerate(zip(images1, attn, Y)):
    for i in range(n):    
        #print('image ',image1.shape,image2.shape)
        #for image1,image2 in enumerate(zip(clip1, att)):
        for j in range(4):    

            image1 = images1[i,j,:,:,:]
            image2 = attn[i,j,:,:,:]
            ax = plt.subplot(2, 4, j + 1)
            plt.imshow(image1)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #ax.set_title(str(y))
    
            ax = plt.subplot(2, 4, j + 1 + 4)
            plt.imshow(image2,cmap=plt.cm.jet)
            #plt.gray()
            plt.colorbar()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(title2)

        plt.show()
    #plt.savefig('results/'+experiment_name+'/plots/att_mask.pdf',dpi=300)
    #plt.show()

def display_data2(array):
    plt.figure(figsize=(20,10 ))
    for clip in array:
        for i in range(16):
                ax = plt.subplot(2,8, i + 1)
                plt.imshow(clip[i])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        plt.show()
    #plt.savefig('results/'+experiment_name+'/plots/att_mask.pdf',dpi=300)
    #plt.show()

def infer_from_model(test_loader, device, args, results_path,train_on_imgs):
    
    mse_valence = 0
    mse_arousal = 0
    ppc_valence = 0
    ppc_arousal = 0
    predictions = []
    true = []

    for x,y in tqdm(test_loader):

        #x = x.to(device)
        #y = y.to(device)
        model = torch.load(results_path+args.experiment_name+'/models/'+ 'model.pth').to(device)
        if train_on_imgs == 'False':
            for i in range(len(x)):
                reps,ds = [],[]
                xx = x[i]
                yy = y[i]
                for clip in xx:
                    print('-',clip.unsqueeze(0).shape)
                    _,rep,_ = model(clip.unsqueeze(0).permute(0,3,1,2))
                    reps.append(rep)
    
                for first, second in zip(reps, reps[1:]):  
    
                    dst =  1-distance.euclidean(first.cpu().detach().numpy(),second.cpu().detach().numpy())
                    ds.append(dst)
                plt.subplot(3,1,1)    
                plt.plot(ds, color='r')
                plt.subplot(3,1,2)
                plt.plot(yy, color='b')
                plt.show()
                print('--------shpe')
        else:
            x = torch.from_numpy(np.array(x)).permute(0,3,1,2).float().to(device)                
            y = torch.from_numpy(np.array(y)).float()#.to(device)                                 

            preds = model(x)

            criterion = nn.MSELoss()
    
            mse_v = criterion(preds[:,0],y[:,0].to(device))
            mse_ar = criterion(preds[:,1],y[:,1].to(device))
    
            ppc_v = pearsonr(preds[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy())[0]
            ppc_ar = pearsonr(preds[:,1].cpu().detach().numpy(),y[:,1].cpu().detach().numpy())[0]
    
            mse_valence += mse_v.detach()
            mse_arousal += mse_ar.detach()
    
            ppc_valence += ppc_v
            ppc_arousal += ppc_ar
    
            predictions.append(preds.cpu().detach().numpy())
            true.append(y.cpu().detach().numpy())

     
    predictions = np.concatenate(predictions)
    true = np.concatenate(true)
    mse_valence = mse_valence / len(test_loader)   
    mse_arousal = mse_arousal / len(test_loader)   
    print(ppc_valence)
    ppc_valence = ppc_valence / len(test_loader)   
    ppc_arousal = ppc_arousal / len(test_loader)   
    
    return true, predictions, mse_valence, mse_arousal, ppc_valence, ppc_arousal
