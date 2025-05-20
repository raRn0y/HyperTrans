import torch
from feature_extractors.features import Features
from utils.mvtec3d_util import *
import numpy as np
import math
import os
import scipy.io as sio
import cv2

class RGBFeatures(Features):

    def add_sample_to_mem_bank(self, sample, domain = 'rgb'):
        #organized_pc = sample[1]
        #organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        #unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        #nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        #
        #unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        #rgb_feature_maps, xyz_feature_maps, _, _, center_idx, _ = self(sample[0],unorganized_pc_no_zeros.contiguous())
        if domain in ['rgb', 'xyz']:
            rgb_feature_maps = self(sample, enhanced = False)
        else:
            rgb_feature_maps = self(sample, enhanced = True, domain = domain)

        #print('test in fea/mul')
        #print(rgb_feature_maps[0].shape)
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        #print(rgb_patch.shape)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        #print(rgb_patch.shape) # 784, 768

        if domain == 'rgb':
            self.patch_lib.append(rgb_patch)
        elif domain == 'xyz':
            self.patch_xyz_lib.append(rgb_patch)
        else:
            self.patch_ift_lib.append(rgb_patch)


    def predict(self, sample, mask, label):
        #organized_pc = sample[1]
        #organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        #unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        #nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        #
        #unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!predict rahter than')
        rgb_feature_maps = self(sample)#[0])#, unorganized_pc_no_zeros.contiguous())

        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        #print('rgb_patch.shape')
        #print(rgb_patch.shape)


        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        #print(rgb_patch.shape)

        self.compute_s_s_map(rgb_patch, rgb_feature_maps[0].shape[-2:], mask, label)#, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)


    def predict_enhance_multi_lib(self, sample, mask, label, depth = 0, K = 5, use_rgb = False, use_xyz = True):
        patch_rgb_lib = self.patch_lib.to(self.device)
        patch_xyz_lib = self.patch_xyz_lib.to(self.device)
        try:
            patch_ift_lib = self.patch_ift_lib.to(self.device)
        except:
            #print('patch_ift_lib not exists')
            pass

        c_label = label.to('cuda:0')

        self.deep_feature_extractor.train()
        features = self.deep_feature_extractor.forward(sample).detach()
        if depth >= 1:
            features = features.clone().detach().reshape(-1, 784, 768)
            with torch.no_grad():
                enhanced_features, pseudo_featuers  = self.deep_feature_extractor.hyper_block(features, patch_rgb_lib.unsqueeze(0))
            alpha = 0.9
            features = alpha * features + (1-alpha) * enhanced_features

        patch = features.reshape(768, -1)
        patch_size = int(math.sqrt(patch.shape[0]))

        original_patch = patch.T

        # Initialize scores
        final_scores = []
        min_vals = []

        # RGB inference if enabled
        if use_rgb:
            #print('\t using RGB')
            patch = self.norm_patch_with_lib(original_patch, 'rgb')
            #print(patch.shape)
            #print(patch_rgb_lib.shape)
            dist = torch.cdist(patch, patch_rgb_lib)
            min_val, min_idx = torch.min(dist, dim=1)
            min_vals.append(min_val)

            s_idx = torch.argmax(min_val)
            s_star = torch.max(min_val)
            m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
            m_star = patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)             
            _, nn_idx = torch.topk(dist, k=K, largest=False, dim=1)
            anomaly_scores = []

            for k in range(K):
                m_star_k = patch_rgb_lib[nn_idx[:, k]]
                w_dist_k = torch.cdist(m_star_k, patch_rgb_lib)
                _, knn_idx_k = torch.topk(w_dist_k, k=3, largest=False)
                m_star_knn_k = torch.linalg.norm(m_test - patch_rgb_lib[knn_idx_k[0, 1:]], dim=1)

                D = torch.sqrt(torch.tensor(patch.shape[1]))
                w_k = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn_k / D)) + 1e-5))
                #anomaly_scores.append(w_k * s_star)
                anomaly_scores.append(s_star)

            final_rgb_score = torch.mean(torch.stack(anomaly_scores), dim=0)
            final_scores.append(final_rgb_score)

        # XYZ inference if enabled
        if use_xyz:
            #print('\t using XYZ')
            patch = self.norm_patch_with_lib(original_patch, 'xyz')
            # Similar steps for the XYZ memory bank
            dist_xyz = torch.cdist(patch, patch_xyz_lib)
            min_val_xyz, min_idx_xyz = torch.min(dist_xyz, dim=1)
            min_vals.append(min_val_xyz)
            s_idx_xyz = torch.argmax(min_val_xyz)
            s_star_xyz = torch.max(min_val_xyz)

            # Anomaly score calculations for XYZ
            anomaly_score_xyz = s_star_xyz  # Here you can add specific calculations for XYZ
            final_scores.append(anomaly_score_xyz)

        # Combine all final scores (e.g., average them)
        combined_score = torch.mean(torch.stack(final_scores), dim=0) if final_scores else torch.tensor(0)

        # Generate segmentation map
        rgb_weight=0.3
        xyz_weight=0.7
        combined_min_val = (rgb_weight * min_vals[0] if use_rgb or use_xyz else 0) + \
                       (xyz_weight * min_vals[1] if use_rgb and use_xyz else 0)
        min_val = combined_min_val
        s_map = min_val.view(1, 1, 28, 28)
        s_map_imgsize = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = s_map_imgsize.clone().cpu()
        s_map = self.blur(s_map)

        label = c_label
        #s = None
        #s_map = seg_output
        #s_map = softmax_output[0,1,:,:]#.detach().cpu().numpy()
        self.image_preds.append(combined_score.cpu())
        self.image_labels.append(label.cpu().detach().numpy())
        self.pixel_preds.extend(s_map_imgsize.flatten().cpu().detach().numpy())
        self.pixel_labels.extend(mask.cpu().flatten().detach().numpy())
        self.predictions.append(s_map_imgsize.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        #self.recon_wo.append(reconstructed_img_wo.cpu().flatten().detach().numpy())
        self.origin.append(sample.cpu().flatten().detach().numpy())


    def predict_enhance(self, sample, mask, label, depth = 0, domain = 'rgb'):
        #organized_pc = sample[1]
        #organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        #unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        #nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        #
        #unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        #rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0], unorganized_pc_no_zeros.contiguous())
        if True:
            #patch_rgb_lib = torch.tensor(self.patch_rgb_lib).to(self.device)
            patch_rgb_lib = self.patch_lib.to(self.device)
            patch_xyz_lib = self.patch_xyz_lib.to(self.device)
            try:
                patch_ift_lib = self.patch_ift_lib.to(self.device)
            except:
                #print('patch_ift_lib not exists')
                pass

        criterion = torch.nn.CrossEntropyLoss()
        c_label = label.to('cuda:0')
        #print(c_label)
        #print(c_label)
        #g = label[1].to('cuda:0')
        if   domain == 'rgb':
            g = torch.tensor(0).unsqueeze(0).to('cuda:0')
        elif domain == 'xyz':
            g = torch.tensor(1).unsqueeze(0).to('cuda:0')

        #print(g.shape)
        #print(sample.shape)
        #sample = sample.to('cuda:0')
        #sample = sample[0].to('cuda:0')
        #features = self(sample)
        self.deep_feature_extractor.train()
        #print(sample.shape)
        features = self.deep_feature_extractor.forward(sample).detach()
        optimizer = torch.optim.Adam(
            self.deep_feature_extractor.domain_classifier.parameters(),
            lr = 1e-4,
            weight_decay = 0
        )


        #features = self.norm_patch_with_lib(features, domain)
        if depth >= 1:
            domain_gradient = None
            #center_idx
            #features = self.deep_feature_extractor(rgb).detach()
            #features = features.reshape(-1, 768*784) 
            #features = features.reshape(-1, 768, 28, 28) 
            #features = features.reshape(-1, 768, 28, 28) 
            #features = features.reshape(-1, 3, 16, 16).clone().detach()
            features = features.clone().detach().reshape(-1, 784, 768)
            #features = features.reshape(-1, 784*3, 16, 16) 
            #features.requires_grad_(True) 
            #features.retain_grad()
            #g_hat_result = self.deep_feature_extractor.domain_classifier(features.view(1, -1))
            #loss_d = criterion(g_hat_result, g)
            #optimizer.zero_grad()
            #loss_d.backward(retain_graph=True)
            #domain_gradient = features.grad
            #print(domain_gradient.shape)
            #features.reshape(-1, 784, 3, 16, 16)
            with torch.no_grad():
                
                #if g.item() == 0:
                #if not 1 in g:
                if   domain == 'rgb':
                    #features = self.deep_feature_extractor.ift(specific, generatic, patch_rgb_lib, patch_xyz_lib)
                    #print('unenhanced feature')
                    #print(features.shape)
                    #print(patch_rgb_lib.unsqueeze(0).shape)
                    #features = self.norm_patch_with_lib(features, 'rgb').reshape(-1, 784, 768)
                    #print(features.shape)
                    enhanced_features, pseudo_featuers  = self.deep_feature_extractor.hyper_block(features, patch_rgb_lib.unsqueeze(0))
                #elif not 0 in g:
                elif domain == 'xyz':
                #else:
                    #features = self.deep_feature_extractor.ift(specific, generatic, patch_xyz_lib, patch_rgb_lib)
                    #print('unenhanced feature')
                    #print(features.shape)
                    #features = self.norm_patch_with_lib(features, 'rgb').reshape(-1, 784, 768)
                    #print(features.shape)
                    enhanced_features, pseudo_featuers  = self.deep_feature_extractor.hyper_block(features, patch_rgb_lib.unsqueeze(0))
                else:
                    exit()
            #features = enhanced_features
            alpha = 0.9
            #features = alpha * features + (1-alpha) * pseudo_featuers
            features = alpha * features + (1-alpha) * enhanced_features

            #alpha = 0.8
            #features = features - (1-alpha) * pseudo_featuers
            #feature = features!!!


        #recon = self.norm_patch_with_lib(features, domain)
        #recon = recon.reshape(-1, 768, 28, 28)
        #recon = self.deep_feature_extractor.upsample(recon)
        #reconstructed_img_wo = self.deep_feature_extractor.image_decoder(recon)

         
        #print('enhanced feature')
        #print(features.shape)
        patch = features.reshape(768, -1)
        #patch = patch.reshape(patch.shape[1], -1).T
        patch_size = int(math.sqrt(patch.shape[0]))

        patch = patch.T
        #print('patch')
        #print(patch.shape)
        #print('patchshape')
        #print(patch.shape)

        #print('domain')
        #print(domain)
        if depth <= 1:
            if domain == 'rgb':
                lib = patch_rgb_lib
            else:
                lib = patch_xyz_lib
        elif depth == 2:
            lib = patch_ift_lib

        #lib = patch_xyz_lib #!!!

        patch = self.norm_patch_with_lib(patch, domain)
        #print(lib.shape)

            # whole patch

        #print(patch)
        dist = torch.cdist(patch, lib) # distance of every small patches of a sample
        #print(dist.shape)
        #print(d)
        #print('dist')
        #print(dist)
        min_val, min_idx = torch.min(dist, dim=1) # min distance, min distance patch
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=3, largest=False)  # pt.2
        m_star_knn = torch.linalg.norm(m_test - lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        #w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star
        # segmentation map
        #print(min_val.shape)
        s_map = min_val.view(1, 1, 28, 28)
        #print('s_map')
        #print(s_map)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')#, align_corners=False)
        s_map = s_map.cpu()
        s_map = self.blur(s_map)
        #print(s_map.shape) # 1, 224, 224

        #patch = feature_maps
        #patch = patch.reshape(patch.shape[1], -1).T

        #patch = (patch - self.xyz_mean)/self.xyz_std
        #patch_lib = self.patch_xyz_lib

        #self.compute_s_s_map(patch, patch_lib, features.shape[-2:], mask, c_label)#, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

        label = c_label
        #s = None
        #s_map = seg_output
        #s_map = softmax_output[0,1,:,:]#.detach().cpu().numpy()
        self.image_preds.append(s.cpu())
        self.image_labels.append(label.cpu().detach().numpy())
        self.pixel_preds.extend(s_map.flatten().cpu().detach().numpy())
        self.pixel_labels.extend(mask.cpu().flatten().detach().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        #self.recon_wo.append(reconstructed_img_wo.cpu().flatten().detach().numpy())
        self.origin.append(sample.cpu().flatten().detach().numpy())

        #self.calculate_metrics()


    def run_coreset(self, domain = 'rgb'):

        if domain == 'rgb':
            self.patch_lib = torch.cat(self.patch_lib, 0)


            self.mean = torch.mean(self.patch_lib)
            self.std = torch.std(self.patch_lib)
            self.patch_lib = (self.patch_lib - self.mean)/self.std

            # self.patch_lib = self.rgb_layernorm(self.patch_lib)

            if self.f_coreset < 1:
                self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                                n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                                eps=self.coreset_eps, )
                self.patch_lib = self.patch_lib[self.coreset_idx]

            #torch.save(self.patch_lib, './results/patch_lib.pt')
            print('source lib saved')
            sio.savemat('patch_lib.mat', {'lib': self.patch_lib.numpy(), 'mean': self.mean, 'std': self.std })


        elif domain == 'xyz':
            self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)

            self.xyz_mean = torch.mean(self.patch_xyz_lib)
            self.xyz_std = torch.std(self.patch_xyz_lib)
            self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

            # self.patch_xyz_lib = self.rgb_layernorm(self.patch_xyz_lib)

            if self.f_coreset < 1:
                self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                                n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                                eps=self.coreset_eps, )
                self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]

            #torch.save(self.patch_xyz_lib, './results/patch_xyz_lib.pt')
            print('target lib saved')
            sio.savemat('patch_xyz_lib.mat', {'lib': self.patch_xyz_lib.numpy(), 'mean': self.xyz_mean, 'std': self.xyz_std })
        else:
            self.patch_ift_lib = torch.cat(self.patch_ift_lib, 0)

            self.ift_mean = torch.mean(self.patch_ift_lib)
            self.ift_std = torch.std(self.patch_ift_lib)
            self.patch_ift_lib = (self.patch_ift_lib - self.ift_mean)/self.ift_std

            # self.patch_ift_lib = self.rgb_layernorm(self.patch_ift_lib)

            if self.f_coreset < 1:
                self.coreset_idx = self.get_coreset_idx_randomp(self.patch_ift_lib,
                                                                n=int(self.f_coreset * self.patch_ift_lib.shape[0]),
                                                                eps=self.coreset_eps, )
                self.patch_ift_lib = self.patch_ift_lib[self.coreset_idx]

            #torch.save(self.patch_ift_lib, './results/patch_ift_lib.pt')
            print('ift lib saved')
            sio.savemat('patch_ift_lib.mat', {'lib': self.patch_ift_lib.numpy(), 'mean': self.ift_mean, 'std': self.ift_std })



    def compute_s_s_map(self, patch, feature_map_dims, mask, label):#, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        patch = (patch - self.mean)/self.std
        print('patch.shape')
        print(patch.shape)

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        print('self.patch_lib.shape')
        print(self.patch_lib.shape)
        dist = torch.cdist(patch, self.patch_lib)
        print('distance')
        print(dist.shape)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        print('min_val.shape')
        print(min_val.shape)
        s_map = min_val.view(1, 1, *feature_map_dims)
        print('s_map.shape')
        print(s_map.shape)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        #print('s')
        #print(s.numpy())
        #print('label')
        #print(label)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

class PointFeatures(Features):

    def add_sample_to_mem_bank(self, sample):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
 
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        self.patch_lib.append(xyz_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        self.compute_s_s_map(xyz_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.args.rm_zero_for_project:
            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib!=0, dim=1))[:,0]]

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]
            
        if self.args.rm_zero_for_project:

            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib!=0, dim=1))[:,0]]
            self.patch_lib = torch.cat((self.patch_lib, torch.zeros(1, self.patch_lib.shape[1])), 0)


    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''


        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

FUSION_BLOCK= True

class FusionFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, rgb_patch2], dim=1)

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_lib.append(fusion_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, rgb_patch2], dim=1)

        self.compute_s_s_map(fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps)
            self.patch_lib = self.patch_lib[self.coreset_idx]

class DoubleRGBPointFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

class DoubleRGBPointFeatures_add(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[s_xyz, s_rgb]])
        s_map = torch.cat([s_map_xyz, s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = s_xyz + s_rgb
        s_map = s_map_xyz + s_map_rgb
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map

class TripleFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        print(sample[0].shape)
        print(sample[1].shape)
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)
        print('rgb patch appended')
        print(rgb_patch.shape)

        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:

            xyz_patch = torch.cat(xyz_feature_maps, 1)
            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
            xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
            xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

            xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

            if FUSION_BLOCK:
                with torch.no_grad():
                    fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
                fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            else:
                fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

            self.patch_xyz_lib.append(xyz_patch)
            print('xyz patch appended')
            print(xyz_patch.shape)
            self.patch_fusion_lib.append(fusion_patch)
    

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    

        self.compute_s_s_map(xyz_patch, rgb_patch, fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    
        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]


        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib!=0, dim=1))[:,0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)


    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

  
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)
 
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
  
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
