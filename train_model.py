import sys
sys.path.append('../')
from utils import *
from data_manager import *
from experiments_manager import *
    
def process_data(arr):
    arr = arr.reshape(arr.shape[0],arr.shape[1],224,224,3)
    data = []
    for clip in arr:
        data.append([np.array(Image.fromarray(img.astype(np.uint8))) for img in clip])
    return data

def to_tensor(data,device): 
    #return torch.stack([torch.Tensor(i) for i in data]).to(device)
    return torch.from_numpy(data).float().to(device)
   
# the used function to include recursivity
def get_rep(X,idx,model,device):
    #print('idx ',idx)
    if idx == 0:
        z = torch.zeros(1, 768).to(device)
        #print(z.shape)
        _,rep,_ = model(X[idx].unsqueeze(0),z,idx)
        #print(rep.shape)
    else:
        _,rep,_ = model(X[idx].unsqueeze(0),get_rep(X,idx-1,model,device),idx)
        #print(X[idx].unsqueeze(0).shape,rep.shape)
    return rep

# the used function to train the model
def train(train_loader,test_loader,args,model1,model2, device,results_path,train_on_imgs):
    if args.mod == 'vis':
        params = list(model1.parameters())
    elif args.mod == 'audio':
        params = list(model2.parameters()) 
    elif args.mod == 'multimodal':
        params = list(model1.parameters()) + list(model2.parameters()) 

    optimizer = optim.Adam(params, lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-6)    
    scheduler = MultiStepLR(optimizer, milestones=[100,200], gamma=0.5)
    criterion = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    m = nn.BatchNorm1d(1280).to(device)
    drop = nn.Dropout(p=0.2)
    pdist = nn.PairwiseDistance(p=2)

    train_loss = 0
    test_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    test_loss1 = 0
    test_loss2 = 0
    tr_tab = []
    te_tab = []

    tr_tab_v = []
    te_tab_v = []

    tr_tab_a = []
    te_tab_a = []

    pcc_tr1 = 0
    pcc_te1 = 0
    pcc_tr2 = 0
    pcc_te2 = 0
    tr_ppc_v = []
    te_ppc_v = []
    tr_ppc_a = []
    te_ppc_a = []

    train_lossb = 0
    train_lossb1 = 0
    train_lossb2 = 0
    pcc_trb1 = 0
    pcc_trb2 = 0
    
    test_lossb = 0
    test_lossb1 = 0
    test_lossb2 = 0
    pcc_teb1 = 0
    pcc_teb2 = 0

    reps = []
    tr_labels = []
    mean1 = 2.80140916252993
    dst1 = 0.6306557158402127
    mean2 = 2.45711663175688
    std2 = 0.954949818254348
    mean = 2.62911719143215
    std = 0.828521483443333    
    feats = []
    y_ds = []
    start = time.time()

    print('[INFO ] start training')
    
    for epoch in tqdm(range(1, args.ep+1)):  
        for x1,x2, y,y_d in train_loader:
        #for x1, y1,l1 in train_loader:
            if train_on_imgs == 'False':
                #x1 = x1.permute(0,2,1,3,4).float().to(device)
                x1 = x1.float().to(device)
                x2 = x2.float().to(device)
                y=torch.stack(y).permute(1,0).float().to(device)
                y = (y - mean) /std
                y_d = y_d.long().to(device)

                optimizer.zero_grad()

                if args.mod == 'vis':
                    output,yd,feat1 = model1(x1)
                elif args.mod == 'audio':
                    output,_,feat1 = model2(x2)
                elif args.mod == 'multimodal':
                    out1,yd,feat1 = model1(x1)
                    out2,_,feat2 = model2(x2)
                    layer_dim = feat1.shape[1]+feat2.shape[1]
                    layer1 = nn.Sequential(nn.LayerNorm(layer_dim), nn.Linear(layer_dim,2)).to(device)
                    feat = torch.cat((feat1,feat2),1)
                    output = layer1(feat)
                    #print(output.shape)
                domain_loss = criterion2(yd ,y_d)
                var_reg = torch.var(torch.cat([torch.mean(output[:,0]).view(1),torch.mean(output[:,1]).view(1)]))
                loss1 = criterion(out1,y)
                loss2 = criterion(out2.float(),y)
                v_loss = criterion(output[:,0],y[:,0])
                a_loss = criterion(output[:,1],y[:,1])

                if args.domain_loss == 'True' : 
                    loss =  criterion(output,y) - args.reg*domain_loss + args.reg*var_reg
                elif args.domain_loss == 'False':
                    loss =  criterion(output,y) + loss1 + loss2 
                                      
                feats.append(feat.cpu().detach())
                y_ds.append(y_d.cpu().detach())

                pcc_v = pearsonr(output[:,0].cpu().detach(),y[:,0].cpu().detach())[0]
                pcc_a = pearsonr(output[:,1].cpu().detach(),y[:,1].cpu().detach())[0]

                loss.backward()
                optimizer.step()

                train_loss += loss.detach()
                train_loss1 += v_loss.detach()
                train_loss2 += a_loss.detach()
                pcc_tr1 += np.arctanh(pcc_v)
                pcc_tr2 += np.arctanh(pcc_a)

            else:
                
                x1 = x1.squeeze()
                y1 = y1.squeeze()
                
                for x,y in zip(x1,y1):
                    y = y.float().unsqueeze(0).to(device)
                    optimizer.zero_grad()
                    output,_ = model1(x.float().unsqueeze(0).to(device))
                    loss = criterion(output,y)
                    loss_v = criterion(output[:,0],y[:,0])
                    loss_a = criterion(output[:,1],y[:,1])
                    pcc_v = pearsonr(output[:,0].cpu().detach(),y[:,0].cpu().detach())[0]
                    pcc_a = pearsonr(output[:,1].cpu().detach(),y[:,1].cpu().detach())[0]
                    train_lossb += loss.detach()
                    train_lossb1 += loss_v.detach()
                    train_lossb2 += loss_a.detach()
                    pcc_trb1 += np.arctanh(pcc_v)
                    pcc_trb2 += np.arctanh(pcc_a)

                    loss.backward()
                    optimizer.step()
                    
                train_loss += train_lossb/l1[0]
                train_loss1 += train_lossb1/l1[0]
                train_loss2 += train_lossb2/l1[0]
                pcc_tr1 += pcc_trb1/l1[0]
                pcc_tr2 += pcc_trb2/l1[0]
                print(l1,train_loss,train_loss1,train_loss2,pcc_tr1,pcc_tr2)

                
        train_loss = train_loss/len(train_loader)
        train_loss1 = train_loss1/len(train_loader)
        train_loss2 = train_loss2/len(train_loader)
        pcc_tr1 = pcc_tr1/len(train_loader)
        pcc_tr2 = pcc_tr2/len(train_loader)

        
        with torch.no_grad():
            i = 0
            for xte1,xte2, yte in test_loader:
            #for xte1, yte,lte in test_loader:

                if train_on_imgs == 'False':

                    xte1 = xte1.float().to(device)
                    xte2 = xte2.float().to(device)
                    yte=torch.stack(yte).permute(1,0).float().to(device)
                    yte = (yte-mean) /std
                   
                    if args.mod == 'vis':
                        output_te,_,feat_te1 = model1(xte1)
                    elif args.mod == 'audio':
                        output_te,_,feat_te2 = model2(xte2)
                    elif args.mod == 'multimodal':
                        out_te1,_,feat_te1 = model1(xte1)
                        out_te2,_,feat_te2 = model2(xte2)
                        #layer_dim = feat1.shape[1]+feat2.shape[1]
                        #layer1 = nn.Sequential(nn.LayerNorm(layer_dim), nn.Linear(layer_dim,2)).to(device)
                        output_te = layer1(torch.cat((feat_te1,feat_te2),1))
                        #print(output_te.shape)
                    pcc1 = pearsonr(output_te[:,0].cpu().detach(),yte[:,0].cpu().detach())[0]
                    pcc1_a = pearsonr(output_te[:,1].cpu().detach(),yte[:,1].cpu().detach())[0]
                   
                    pcc_te1 += np.arctanh(pcc1)
                    pcc_te2 += np.arctanh(pcc1_a)

                    loss_te = criterion(output_te,yte)
                    l_v = criterion(output_te[:,0],yte[:,0])
                    l_a = criterion(output_te[:,1],yte[:,1])
                    test_loss += loss_te.detach()
                    test_loss1 += l_v.detach()
                    test_loss2 += l_a.detach()


                else:
                    xte1 = xte1.squeeze()
                    yte1 = yte.squeeze()

                    for xte,yte in zip(xte1,yte1):
                        
                        output_te,_ = model1(xte.float().unsqueeze(0).to(device))
                        yte = yte.float().unsqueeze(0).to(device)
                        loss_te = criterion(output_te,yte)
                        loss_v_te = criterion(output_te[:,0],yte[:,0])
                        loss_a_te = criterion(output_te[:,1],yte[:,1])
                        
                        pcc_v_te = pearsonr(output_te[:,0].cpu().detach(),yte[:,0].cpu().detach())[0]
                        pcc_a_te = pearsonr(output_te[:,1].cpu().detach(),yte[:,1].cpu().detach())[0]

                        test_lossb += loss_te.detach()
                        test_lossb1 += loss_v_te.detach()
                        test_lossb2 += loss_a_te.detach()
                        pcc_teb1 += np.arctanh(pcc_v_te)
                        pcc_teb2 += np.arctanh(pcc_a_te)
                        
                    test_loss += test_lossb/lte[0]
                    test_loss1 += test_lossb1/lte[0]
                    test_loss2 += test_lossb2/lte[0]
                    pcc_te1 += pcc_teb1/lte[0]
                    pcc_te2 += pcc_teb2/lte[0]
                    
            test_loss = test_loss/len(test_loader)
            test_loss1 = test_loss1/len(test_loader) 
            test_loss2 = test_loss2/len(test_loader)
            pcc_te1 = pcc_te1/len(test_loader)
            pcc_te2 = pcc_te2/len(test_loader)
        
        scheduler.step()
        print('epoch ',epoch, 'training loss ',train_loss, train_loss1,train_loss2)
        print('-------------------------------------------------------------------------')
        print('epoch ',epoch, 'training pcc ',np.tanh(pcc_tr1),np.tanh(pcc_tr2))
        print('-----------------------------------------------------------------------------------------')
        print('epoch ',epoch, 'testing loss ',test_loss, test_loss1, test_loss2)
        print('-----------------------------------------------------------------------------------------')
        print('epoch ',epoch, 'testing pcc ',np.tanh(pcc_te1),  np.tanh(pcc_te2))
        print('-----------------------------------------------------------------------------------------')

        
        tr_tab.append(train_loss.cpu().detach())
        te_tab.append(test_loss.cpu().detach())

        tr_tab_v.append(train_loss1.cpu().detach())
        te_tab_v.append(test_loss1.cpu().detach())

        tr_tab_a.append(train_loss2.cpu().detach())
        te_tab_a.append(test_loss2.cpu().detach())

        tr_ppc_v.append(pcc_tr1)
        te_ppc_v.append(pcc_te1)
        tr_ppc_a.append(pcc_tr2)
        te_ppc_a.append(pcc_te2)        
        
    print('[INFO ] training finished, ellapsed time ',time.time()-start)    
    # save the model
    model_folder = os.mkdir(results_path+'/'+args.experiment_name+'/models')
    model_path = results_path+'/'+args.experiment_name+'/models/'+ 'model.pth'
    torch.save(model1, model_path)

    plt.figure(1)
    plt.plot(tr_tab,color='r',label='train loss')
    plt.plot(te_tab,color='b',label='test loss')
    val, idx = min((val, idx) for (idx, val) in enumerate(te_tab))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val - 0.01),arrowprops=dict(facecolor='black', shrink=0.0005))
    plt.legend()
    plt.savefig(results_path+'/'+args.experiment_name+'/plots/loss.pdf',dpi=300)
    plt.close()

    plt.figure(4)
    plt.plot(tr_tab_v,color='r',label='train loss')
    plt.plot(te_tab_v,color='b',label='test loss')
    val, idx = min((val, idx) for (idx, val) in enumerate(te_tab_v))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val - 0.01),arrowprops=dict(facecolor='black', shrink=0.0005))
    plt.legend()
    plt.savefig(results_path+'/'+args.experiment_name+'/plots/loss-v.pdf',dpi=300)
    plt.close()

    plt.figure(5)
    plt.plot(tr_tab_a,color='r',label='train loss')
    plt.plot(te_tab_a,color='b',label='test loss')
    val, idx = min((val, idx) for (idx, val) in enumerate(te_tab_a))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val - 0.01),arrowprops=dict(facecolor='black', shrink=0.0005))   
    plt.legend()
    plt.savefig(results_path+'/'+args.experiment_name+'/plots/loss-a.pdf',dpi=300)
    plt.close()
    
    plt.figure(2)
    plt.plot(tr_ppc_v,color='r',label='train ppc-v')
    plt.plot(te_ppc_v,color='b',label='test ppc-v')
    val, idx = max((val, idx) for (idx, val) in enumerate(te_ppc_v))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val - 0.01),arrowprops=dict(facecolor='black', shrink=0.0005))
    plt.legend()
    plt.savefig(results_path+'/'+args.experiment_name+'/plots/ppc-v.pdf',dpi=300)
    plt.close()

    plt.figure(3)
    plt.plot(tr_ppc_a,color='r',label='train ppc-a')
    plt.plot(te_ppc_a,color='b',label='test ppc-a')
    val, idx = max((val, idx) for (idx, val) in enumerate(te_ppc_a))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val - 0.01),arrowprops=dict(facecolor='black', shrink=0.0005))
    plt.legend()
    plt.savefig(results_path+'/'+args.experiment_name+'/plots/ppc-a.pdf',dpi=300)
    plt.close()

    visualize_SNE(np.concatenate(np.array(feats),axis=0),np.concatenate(np.array(y_ds),axis=0),30,1000,160,results_path+'/'+args.experiment_name+'/plots/TSNE.pdf')                                                                                            

    '''
    for first, second in zip(reps, reps[1:]):
        #sp_corr.append(cosine_similarity(first,second))
        ssim = stats.spearmanr(first,second)
        #print(ssim)
        corr.append(ssim[0])           

    plt.figure(2)
    plt.plot(corr)
    plt.savefig('corr.pdf')
    plt.close()

    print(stats.spearmanr(corr,tr_labels[1:]))
    #print(stats.spearmanr(corr,tr_labels[:,1]))
    '''
