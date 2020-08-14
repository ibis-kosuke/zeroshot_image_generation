from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2, build_images
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET, G_NET_not_CA, Att_Classifier
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys
import pickle as pkl

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, args, output_dir, data_loader):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')

        self.gpus = list(map(int, args.gpu_ids.split(",")))
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.args = args
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.display_interval = 50

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.sample_num = 100//self.batch_size
        self.att_num = self.load_att_num() ###いらない。
        

    def load_att_num(self):
        data_dir = cfg.DATA_DIR
        att_num_path = os.path.join(data_dir, "att_num.pickle")
        with open(att_num_path, "rb") as f:
            att_num = pkl.load(f)
        att_num += 1 ##for can not classifying 
        return att_num


    def build_models(self):
        ##feature extractor
        inception_model = CNN_ENCODER()

        ## classifier networks
        classifiers = []
        for i in range(cfg.ATT_NUM):
            cls_model = Att_Classifier()
            classifiers.append(cls_model)

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            if not self.args.kl_loss:
                netG = G_NET_not_CA(self.args) 
            else:
                netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            netG.cuda()
            inception_model.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
            for i in range(len(classifiers)):
                classifiers[i].cuda()
    
        return [netG, netsD, inception_model, classifiers, epoch]

    def define_optimizers(self, netG, netsD, classifiers):
        optimizersC = []
        for i in range(len(classifiers)):
            opt = optim.Adam(classifiers[i].parameters(), lr=cfg.TRAIN.C_LR,
                                betas=(0.5,0.999))
            optimizersC.append(opt)

        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
            
        return optimizerG, optimizersD, optimizersC

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()

        return real_labels, fake_labels

    def save_model(self, netG, avg_param_G, netsD, classifiers, epoch):
        mkdir_p(self.model_dir)

        for i in range(len(classifiers)):
            classifier = classifiers[i]
            torch.save(classifier.state_dict(), 
                        '%s/classifier_%d.pth' %(self.model_dir, i))

        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds/classifiers models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires


    def save_img_results(self, netG, noise, atts, image_att, inception_model, classifiers, imgs, gen_iterations, name='current'):
        mkdir_p(self.image_dir)

        # Save images
        if self.args.kl_loss:
            fake_imgs, _, _ = netG(noise, atts) ##
        else:
            fake_imgs, _  = netG(noise, atts, image_att, inception_model, classifiers, imgs)

        ##make image set
        img_set = build_images(fake_imgs)##
        img = Image.fromarray(img_set)
        full_path = '%s/G_%s.png' % (self.image_dir, gen_iterations)
        img.save(full_path)


    def train(self):
        netG, netsD, inception_model, classifiers, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD, optimizersC = self.define_optimizers(netG, netsD, classifiers)
        real_labels, fake_labels= self.prepare_labels()
        writer = SummaryWriter(self.args.run_dir)

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            print("epoch: {}/{}".format(epoch, self.max_epoch))
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                if(step%self.display_interval==0):
                    print("step:{}/{} {:.2f}%".format(step, self.num_batches, step/self.num_batches*100))
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, atts, image_atts, class_ids, keys = prepare_data(data)

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                

                """
                if not self.args.kl_loss:
                    fake_imgs, C_losses = nn.parallel.data_parallel( netG, (noise, atts, image_atts,
                                                 inception_model, classifiers, imgs), self.gpus)
                else:
                    fake_imgs, mu, logvar = netG(noise, atts, image_atts) ## model内の次元が合っていない可能性。
                """


                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                errD_dic={}
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              atts, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())
                    errD_dic['D_%d'%i] = errD.item()

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errC_total = 0
                C_logs = ''
                for i in range(len(classifiers)):
                    classifiers[i].zero_grad()
                    C_losses[i].backward()
                    optimizersC[i].step()
                    errC_total += C_losses[i]
                C_logs += 'errC_total: %.2f ' % ( errC_total.item())

                for i in range(netG.parameters()):
                    if i==0:
                        print(netG.parameters()[i].grad)

                 ##TODO netGにgradientが溜まっているかどうかを確認せよ。

                errG_total = 0
                errG_total, G_logs, errG_dic = \
                    generator_loss(netsD, fake_imgs, real_labels, atts, errG_total)
                if self.args.kl_loss:
                    kl_loss = KL_loss(mu, logvar)
                    errG_total += kl_loss
                    G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                    writer.add_scalar('kl_loss', kl_loss.item(), epoch*self.num_batches+step)
                
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for i in range(len(optimizersC)):
                    optimizersC[i].step()

                errD_dic.update(errG_dic)
                writer.add_scalars('training_losses', errD_dic, epoch*self.num_batches+step)

                """ 
                self.save_img_results(netG, fixed_noise, atts,image_atts,
                                          epoch, name='average') ##for debug
                """
                                          
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs +'\n' + C_logs)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, atts, image_atts, inception_model, classifiers, imgs,
                                          epoch, name='average')
                    load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Loss_C: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(), errC_total.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, classifiers, epoch)

        self.save_model(netG, avg_param_G, netsD, classifiers, self.max_epoch)


    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)


    def sampling(self):
        if self.args.netG == '':
            print('Error: the path for models is not found!')
        else:
            data_dir = cfg.DATA_DIR
            if self.args.split == "test_unseen":
                filepath = os.path.join(data_dir, "test_unseen/class_data.pickle")
            else: #test_seen
                filepath = os.path.join(data_dir, "test_seen/class_data.pickle")
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    data_dic = pkl.load(f)
            class_names = data_dic['classes']
            class_ids = data_dic['class_info']

            att_dir = os.path.join(data_dir,"CUB_200_2011/attributes")
            att_np = np.zeros((312, 200)) #for CUB
            with open(att_dir+"/class_attribute_labels_continuous.txt", "r") as f:
                for ind, line in enumerate(f.readlines()):
                    line = line.strip("\n")
                    line = list(map(float, line.split()))
                    att_np[:,ind] = line

            if self.args.kl_loss:
                netG = G_NET()
            else:
                netG = G_NET_not_CA()
            test_model = "netG_epoch_600.pth"
            model_path = os.path.join(self.args.netG, "Model", test_model) ##
            state_dic = torch.load(model_path, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dic)
            netG.cuda()
            netG.eval()

            noise = torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM)
            
            for class_name, class_id in zip(class_names,class_ids):
                print("now generating, ", class_name)
                class_dir = os.path.join(self.args.netG, 'valid', test_model[:test_model.rfind(".")],self.args.split, class_name)
                atts = att_np[:,class_id-1]
                atts = np.expand_dims(atts, axis=0)
                atts = atts.repeat(self.batch_size, axis=0)
                assert atts.shape==(self.batch_size,312)

                if cfg.CUDA:
                    noise = noise.cuda()
                    atts = torch.cuda.FloatTensor(atts)
                else:
                    atts = torch.FloatTensor(atts)

                for i in range(self.sample_num):
                    noise.normal_(0,1)
                    if self.args.kl_loss:
                        fake_imgs, _ ,_ = nn.parallel.data_parallel(netG, (noise, atts), self.gpus)
                    else:
                        fake_imgs = nn.parallel.data_parallel(netG, (noise, atts), self.gpus)
                    for stage in range(len(fake_imgs)):
                        for num,im in enumerate(fake_imgs[stage]):
                            im = im.detach().cpu()
                            im = im.add_(1).div_(2).mul_(255)
                            im = im.numpy().astype(np.uint8)
                            im = np.transpose(im, (1,2,0))
                            im = Image.fromarray(im)
                            stage_dir = os.path.join(class_dir, "stage_%d" % stage)
                            mkdir_p(stage_dir)
                            img_path = os.path.join(stage_dir, "single_%d.png" % num)
                            im.save(img_path)
                        for j in range(int(self.batch_size/20)): ## cfg.batch_size==100
                            one_set = [fake_imgs[0][j*20:(j+1)*20],fake_imgs[1][j*20:(j+1)*20], fake_imgs[2][j*20:(j+1)*20]]
                            img_set = build_images(one_set)
                            img_set = Image.fromarray(img_set)
                            super_dir = os.path.join(class_dir, "super")
                            mkdir_p(super_dir)
                            img_path = os.path.join(super_dir, "super_%d.png" % j)
                            img_set.save(img_path)
                            


    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
