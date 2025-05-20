import torch
from tqdm import tqdm
import os
from feature_extractors import multiple_features
from utils.dataset import get_data_loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.au_pro_util import calculate_au_pro
import math
import scipy.io as sio



class M3DM():
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size
        self.count = args.max_sample
        if args.method_name == 'DINO':
            self.methods = {
                "DINO": multiple_features.RGBFeatures(args),
            }
        elif args.method_name == 'Point_MAE':
            self.methods = {
                "Point_MAE": multiple_features.PointFeatures(args),
            }
        elif args.method_name == 'Fusion':
            self.methods = {
                "Fusion": multiple_features.FusionFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE':
            self.methods = {
                "DINO+Point_MAE": multiple_features.DoubleRGBPointFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE+add':
            self.methods = {
                "DINO+Point_MAE": multiple_features.DoubleRGBPointFeatures_add(args),
            }
        elif args.method_name == 'DINO+Point_MAE+Fusion':
            self.methods = {
                "DINO+Point_MAE+Fusion": multiple_features.TripleFeatures(args),
            }


    def fit(self, class_name, domain = 'rgb'):
        if domain == 'rgb':
            train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size, args=self.args)
        elif domain == 'xyz':
            train_loader = get_data_loader("train_fewshot", class_name=class_name, img_size=self.image_size, args=self.args)

        flag = 0

        samples = []
        print('ift source not specified')
        for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            for method in self.methods.values():
                if   domain == 'rgb':
                    sample = sample[0].to('cuda:0')
                elif domain == 'xyz':
                    sample = sample[2].to('cuda:0')
                else:
                    sample = sample[2].to('cuda:0')
                samples.append(sample.cpu().squeeze(0))
                if self.args.save_feature:
                    method.add_sample_to_mem_bank(sample, class_name=class_name, domain = domain)
                else:
                    method.add_sample_to_mem_bank(sample, domain = domain)
                flag += 1
            if flag > self.count:
                flag = 0
                break
                

        method.save_path = 'results/'+class_name+'/'+domain+"_"

        for method_name, method in self.methods.items():
            print(f'Running coreset for {method_name} on class {class_name}...')
            method.run_coreset(domain = domain)
            

    def evaluate(self, class_name, depth = 0, domain = 'rgb', save = False):
        if depth == 0:
            print('\n\tTesting NOT Enhanced........................................')
            #enhance = True
        elif depth == 1:
            print('\n\tTesting Enhanced....................................')
            #enhance = False
        elif depth == 2:
            print('\n\tTesting Enhanced with NEW Banks....................................')

        if domain == 'rgb':
            print('\tusing RGB')
        elif domain == 'xyz':
            print('\tusing XYZ')
        else:
            print('\tusing MUl')

        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)
        path_list = []
        
        for sample, mask, label, rgb_path in tqdm(test_loader, desc=f'........Extracting test features for class {class_name}'):
            sample = sample[2].to('cuda:0') # inference with xyz

            for method in self.methods.values():
                method.save_path = 'results/'+class_name+'/level'+str(depth)+'_'+domain+'_'
                if domain == 'rgb':
                    method.predict_enhance_multi_lib(sample, mask, label, depth = depth, use_rgb = True, use_xyz = False)
                elif domain == 'xyz':
                    method.predict_enhance_multi_lib(sample, mask, label, depth = depth, use_rgb = False, use_xyz = True)
                else:
                    method.predict_enhance_multi_lib(sample, mask, label, depth = depth, use_rgb = True, use_xyz = True)
                path_list.append(rgb_path)
                        

        for method_name, method in self.methods.items():
            method.calculate_metrics(path_list, save = save)
            image_rocaucs[method_name] = round(method.image_rocauc, 3)
            pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3)
            au_pros[method_name] = round(method.au_pro, 3)
            print(
                f'\tClass: {class_name}, {method_name} Image ROCAUC: {method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {method.pixel_rocauc:.3f}, {method_name} AU-PRO: {method.au_pro:.3f}')
        return image_rocaucs, pixel_rocaucs, au_pros

    #def learn_reconstruct_and_segment(self, classes, domain):
    def learn_reconstruct_and_segment(self, path_dir, epochs, cls, domain, args=None):
        train_loader = get_data_loader("train", class_name=cls, img_size=args.img_size, args=args )

        method = self.methods['DINO']

        patch_rgb_lib = self.methods['DINO'].patch_lib.to('cuda:0')

        model = self.methods['DINO'].deep_feature_extractor
        optimizerD = torch.optim.Adam(
            model.domain_classifier.parameters(),
            #momentum = 0.9,
            lr = 1e-4,
            weight_decay = 0
        )
        optimizerC = torch.optim.Adam(
            [
                {'params':model.hyper_block.parameters()},
            ],
            lr = 1e-2,
            weight_decay = 0
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerC,[200, 600],gamma=0.5, last_epoch=-1)

        criterion = torch.nn.CrossEntropyLoss()
        total_iters = len(train_loader)
        for epo in range(epochs):
            pixel_labels = []
            pixel_preds = []
            image_labels = []
            image_preds = []

            recons = []
            print(f'\n\nepoch {epo} of Learning Reconstruction and Segmentation..........................')

            # path setting
            save_path = os.path.join(path_dir , 'epoch_'+ str(epo) + '_')
            print(save_path)
            #method.save_path = save_path

            with tqdm(train_loader, desc="Reconstruct") as pbar:
                #for sample, y, i in pbar:
                for sample, y in pbar:#tqdm(train_loader, desc=f'Extracting'):

                    model.train()
                    if domain == 'rgb':
                        x_wo = sample[0].to('cuda:0')
                        g = torch.tensor(0).unsqueeze(0).to('cuda:0')
                    else:
                        x_wo = sample[2].to('cuda:0')
                        g = torch.tensor(1).unsqueeze(0).to('cuda:0')
                    c = y[0].to('cuda:0')
                    domain_gradient = None
                    features = model.forward(x_wo).detach()#.flatten()

                    features = features.clone().detach().reshape(-1, 784, 768)

                    enhanced_features, pseudo_featuers  = model.hyper_block(features, patch_rgb_lib.unsqueeze(0))
                    # 112 * 112 
                    loss_l2 = torch.nn.functional.mse_loss(enhanced_features, features)# + torch.nn.functional.mse_loss(enhanced_features, pseudo_featuers)

                    optimizerC.zero_grad()
                    loss_C = loss_l2 #+ loss_disturb
                    loss_C.backward()
                    optimizerC.step()
                    pbar.set_postfix({
                        'Loss_l2     ': '{:.4f}'.format(loss_l2.item()),
                    })
                    pbar.update(1)

                # One batch ends
            # One epoch ends
            scheduler.step()
            #if True or epo % 10 == 0:
            if epo % 10 == 0:
                image_rocaucs, pixel_rocaucs, au_pros = self.evaluate(cls, depth = 1, domain = 'xyz', save=True)#, domain = 'rgb')
                image_rocaucs, pixel_rocaucs, au_pros = self.evaluate(cls, depth = 1, domain = 'rgb', save=True)#, domain = 'rgb')
                image_rocaucs, pixel_rocaucs, au_pros = self.evaluate(cls, depth = 1, domain = 'multi', save=True)#, domain = 'rgb')
                try:
                    if not os.path.exists(path_dir+"checkpoints/"):
                        os.makedirs(path_dir+"checkpoints/")
                    torch.save(model.state_dict(), path_dir+"checkpoints/"+"model.pckl")

                    sio.savemat('recons.mat', {'recons': np.array(recons)})
                except:
                    print('WARNING: failed on m3dm_runner/saving')

    def learn_disturb(self, path_dir, epochs, cls, domain, args=None):
        train_loader = get_data_loader("train", class_name=cls, img_size=args.img_size, args=args)
        method = self.methods['DINO']
        patch_rgb_lib = self.methods['DINO'].patch_lib.to('cuda:0')
        model = self.methods['DINO'].deep_feature_extractor
        optimizerC = torch.optim.Adam(
            [
                {'params': layer.parameters() for layer in model.layers[3:model.num_disturb]}
            ],
            lr = 1e-6,
            weight_decay = 0
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerC,[200, 600],gamma=0.5, last_epoch=-1)

        criterion = torch.nn.CrossEntropyLoss()
        total_iters = len(train_loader)

        for epo in range(epochs):
            pixel_labels = []
            pixel_preds = []
            image_labels = []
            image_preds = []

            recons = []
            print(f'\n\nepoch {epo} of Disturbance Learning..........................')
            # path setting
            save_path = os.path.join(path_dir , 'epoch_'+ str(epo) + '_')
            print(save_path)

            with tqdm(train_loader, desc="Reconstruct") as pbar:
                for sample, y in pbar:#tqdm(train_loader, desc=f'Extracting'):

                    model.train()

                    x_wo = sample[0].to('cuda:0') # get rgb
                    g = torch.tensor(0).unsqueeze(0).to('cuda:0')
                    c = y[0].to('cuda:0')
                    domain_gradient = None
                    features = model.forward(x_wo).detach()#.flatten()
                    features_disturbed = model.forward(x_wo, disturb = True).reshape(-1, 784, 768)#.flatten()

                    features = features.clone().detach().reshape(-1, 784, 768)
                    enhanced_features, _  = model.hyper_block(features, patch_rgb_lib.unsqueeze(0))
                    loss_disturb = torch.nn.functional.mse_loss(features_disturbed, features)# + torch.nn.functional.mse_loss(enhanced_features, pseudo_featuers)

                    optimizerC.zero_grad()
                    loss_C = loss_disturb
                    loss_C.backward()
                    optimizerC.step()
                    pbar.set_postfix({
                        'Loss_disturb': '{:.4f}'.format(loss_disturb.item()),
                    })
                    pbar.update(1)
            scheduler.step()
            if True:
                #image_rocaucs, pixel_rocaucs, au_pros = self.evaluate(cls, depth = 1, domain = 'xyz')#, domain = 'rgb')
                image_rocaucs, pixel_rocaucs, au_pros = self.evaluate(cls, depth = 1, domain = 'xyz')#, domain = 'rgb')
                image_rocaucs, pixel_rocaucs, au_pros = self.evaluate(cls, depth = 1, domain = 'rgb')#, domain = 'rgb')
                image_rocaucs, pixel_rocaucs, au_pros = self.evaluate(cls, depth = 1, domain = 'multi')#, domain = 'rgb')

                try:
                    if not os.path.exists(path_dir+"checkpoints/"):
                        os.makedirs(path_dir+"checkpoints/")
                    torch.save(model.state_dict(), path_dir+"checkpoints/"+"model.pckl")

                    sio.savemat('recons.mat', {'recons': np.array(recons)})
                except:
                    print('WARNING: failed on m3dm_runner/saving')


