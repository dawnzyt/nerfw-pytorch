'''
shared train_dataset, valid nerf version of DDP
'''

import os.path
import os
import pickle
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset.cambridge import CambridgeDataset
from dataset.seven_scenes import SevenScenesDataset
# from model.nerfw_system import NeRFWSystem       # old  version
from model.nerfw_system_updated import NeRFWSystem # latest version
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.evaluation_metrics import psnr
from option.nerf_option import NeRFOption
from utils.data_loger import DataLoger
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.utils import visualize_depth
import sys

# 初始化分布式环境
def init_processes(rank, size, backend='nccl'):
    if sys.platform == "win32":
        backend = 'gloo'
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:23333',  # 初始化方法，tcp初始化需要指定一个地址
        rank=rank,
        world_size=size
    )

    # 確保每個進程都能使用相同的隨機種子
    seed = 0
    torch.manual_seed(seed)

    # 設置每個進程使用的設備
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)


def trainer(rank, world_size, opt, shared_train, shared_valid):
    # 初始化分布式环境
    init_processes(rank, world_size)
    print('GPU:{} start...'.format(rank))

    # init sampler and dataloader
    train_dataset = shared_train.dataset
    valid_dataset = shared_valid.dataset
    train_sampler = DistributedSampler(train_dataset, shuffle=True)  # DistributedSampler保证各进程的batch samples来自一个子集。
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=0, sampler=train_sampler)
    loger = DataLoger(root_dir=opt.root_dir, exp_name=opt.exp_name)
    nerf = NeRFWSystem(train_dataset.N_views, opt.N_c, opt.N_f, opt.use_disp,
                       opt.perturb, opt.layers, opt.W, opt.N_xyz_freq, opt.N_dir_freq, opt.encode_a, opt.encode_t,
                       opt.a_dim, opt.t_dim, beta_min=opt.beta_min, lambda_u=opt.lambda_u)
    nerf = nerf.to(rank)
    optimizer = Adam(nerf.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    steps, epoch_steps, valid_times = 0, 0, 0  #
    if opt.last_epoch:  # >0: resume
        # resume ckpt: epoch_steps, nerf model and optimizer
        ckpt = torch.load(opt.ckpt_path, map_location=f'cuda:{rank}')
        nerf.load_state_dict(ckpt['nerf'])  # cuda:0 load model
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch_steps = ckpt['epoch_steps']
        steps += (opt.last_epoch - 1) * len(dataloader) + epoch_steps
        print(f'cuda:{rank} resumed to epoch{opt.last_epoch}, epoch_steps{epoch_steps}')

    nerf = DDP(nerf, device_ids=[rank], output_device=rank, find_unused_parameters=True)  # DDP: distributed

    # start training
    start_epoch = opt.last_epoch if (epoch_steps != len(dataloader) and epoch_steps) else opt.last_epoch + 1
    for epoch in range(start_epoch, opt.epochs + 1):
        if epoch != opt.last_epoch:  # continue training
            epoch_steps = 0
        c_psnrs = []
        f_psnrs = []
        for rays in tqdm(dataloader, initial=epoch_steps, total=len(dataloader), file=sys.stdout):
            epoch_steps += 1
            steps += 1
            rays = rays.to(rank)
            # 这里的result以及loss实际上还是单个进程出来的。DDP注册的model会在每个参数上注册一个autograd hook，当相应的梯度在后向传播中计算出来时，这个hook就会触发，然后调用分布式通信接口来进行梯度的all_reduce, 即在注册了DDP model的optimizer 在step之前, DDP会自动调用barrier来等待所有进程的梯度all_reduce完成
            res_c, res_f, losses = nerf.forward(rays[:, :9], rays[:, 9:], cal_loss=True)
            c, fl, fr, c_psnr, f_psnr = losses['coarse'], losses['fine'], losses['fine_regular'], losses['c_psnr'], \
                losses['f_psnr']
            loss = c + fl + (fr if fr is not None else 0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # step前grad DPP實現自動all_reduce, 但loss仍是单进程

            # all_reduce所有进程loss
            # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            # dist.all_reduce(c, op=dist.ReduceOp.SUM)
            # dist.all_reduce(fl, op=dist.ReduceOp.SUM)
            # # dist.all_reduce(fr, op=dist.ReduceOp.SUM)
            # dist.all_reduce(c_psnr, op=dist.ReduceOp.SUM)
            # dist.all_reduce(f_psnr, op=dist.ReduceOp.SUM)
            loss, c, fl, c_psnr, f_psnr = loss / world_size, c / world_size, fl / world_size, c_psnr / world_size, f_psnr / world_size
            fr = (fr / world_size) if fr is not None else None
            c_psnrs.append(c_psnr.item())
            f_psnrs.append(f_psnr.item())
            ########################################
            ############## log loss ################
            if steps % opt.log_freq == 0 and rank == 0:
                losses = {'coarse_loss': c.item(), 'fine_loss': (fl + (fr if fr is not None else 0)).item(),
                          'c_psnr': c_psnr.item(), 'f_psnr': f_psnr.item()}
                loger.log_loss(losses, epoch, steps, epoch_steps)
                print('\n[train] [epoch:%d, steps:%d] [loss] coarse:%.4f fine:%.4f regular:%.4f c_psnr:%.2f f_psnr:%.2f]'
                      % (epoch, epoch_steps, c, fl, (fr if fr is not None else 0), c_psnr, f_psnr))
            ########################################
            ############# checkpoint ###############
            if steps % opt.save_latest_freq == 0 and rank == 0:  # GPU:0 valid
                valid_times += 1
                # valid first
                all_rays = valid_dataset.__getitem__(0)
                all_rays = all_rays.to(rank)
                with torch.no_grad():
                    res = nerf.module.inference(all_rays[:, :9], opt.chunk)
                    fs, ft = res['fs'], res['ft']
                    cc, fc, z = res['cc'], res['fc'], res['z']
                w, h = valid_dataset.downscale_size[0], valid_dataset.downscale_size[1]
                y_img = all_rays[:, 9:].view(h, w, 3)  # GT
                cc = cc.view(h, w, 3)  # coarse img
                fs = fs.view(h, w, 3)  # static img
                ft = ft.view(h, w, 3)  # transient img
                fc = fc.view(h, w, 3)  # fine img = static + transient
                z = z.view(h, w, 1)  # depth
                img_psnr_c, img_psnr_f = psnr(cc, y_img), psnr(fc, y_img)
                psnr_c, psnr_f = float(np.mean(c_psnrs)), float(np.mean(f_psnrs))
                print('[valid] [epoch:%d, steps:%d] [img_psnr] coarse:%.3f fine:%.3f] [mean_psnr] coarse:%.3f fine:%.3f'
                      % (epoch, epoch_steps, img_psnr_c, img_psnr_f, psnr_c, psnr_f))
                loger.log_loss({'coarse_psnr': psnr_c, 'fine_psnr:': psnr_f}, epoch, steps, epoch_steps)

                #########################################################
                ############## visualize and save plt figure ############
                ##############        default encode_t       ############
                c_img = (np.clip(cc.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                fs_img = (np.clip(fs.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                ft_img = (np.clip(ft.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                f_img = (np.clip(fc.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                z = z.cpu().numpy()
                y_img_np = y_img.cpu().numpy()
                # show img in the save plt window
                plt.figure(figsize=(16, 12))
                plt.subplot(3, 2, 1), plt.imshow(y_img_np), plt.title('GT')
                plt.subplot(3, 2, 2), plt.imshow(c_img), plt.title('coarse')
                plt.subplot(3, 2, 3), plt.imshow(fs_img), plt.title('static')
                plt.subplot(3, 2, 4), plt.imshow(ft_img), plt.title('transient')
                plt.subplot(3, 2, 5), plt.imshow(f_img), plt.title('fine')
                plt.subplot(3, 2, 6), plt.imshow(visualize_depth(z)), plt.title('static depth')
                plt.suptitle('coarse_psnr:%.3f, fine_psnr:%.3f' % (img_psnr_c, img_psnr_f))
                plt.savefig(os.path.join(opt.root_dir, opt.exp_name, 'epoch%d_iter%d_psnr-%.3fc-%.3ff.png' %
                                         (epoch, epoch_steps, img_psnr_c, img_psnr_f)))

                #########################################################
                #################### save checkpoints ###################
                ckpt = {'nerf': nerf.module.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch_steps': epoch_steps}
                torch.save(ckpt, os.path.join(opt.root_dir, opt.exp_name, 'ckpt_epoch%d_iter%d_psnr-%.3fc-%.3ff.pkl' % (
                    epoch, epoch_steps, psnr_c, psnr_f)))
            if epoch_steps == len(dataloader):
                break
        scheduler.step()


if __name__ == '__main__':
    opt = NeRFOption().into_opt(save_opt=True)
    world_size = opt.num_gpus
    # shared_dataset:
    shared_train = mp.Manager().Namespace()  # 在多进程之间创建共享命名空间
    shared_valid = mp.Manager().Namespace()
    shared_train.dataset = CambridgeDataset(opt.data_root_dir, opt.scene, split='train',
                                            img_downscale=opt.img_downscale, use_cache=opt.use_cache,
                                            if_save_cache=opt.if_save_cache)
    shared_valid.dataset = CambridgeDataset(opt.data_root_dir, opt.scene, split='valid',
                                            img_downscale=opt.img_downscale, use_cache=opt.use_cache,
                                            if_save_cache=False)
    mp.spawn(trainer, args=(world_size, opt, shared_train, shared_valid), nprocs=world_size, join=True)
