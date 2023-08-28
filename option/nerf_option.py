import argparse
import os.path
import pickle


class NeRFOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic
        self.parser.add_argument('--root_dir', type=str, default="./runs/nerf", help='root dir of exp')
        self.parser.add_argument('--exp_name', type=str, default='test')
        self.parser.add_argument('--Description', type=str, default='')
        self.parser.add_argument('--is_train', type=bool, default=True)

        # train
        self.parser.add_argument('--batch_size', type=int, default=1024)
        self.parser.add_argument('--chunk', type=int, default=1024)
        self.parser.add_argument('--epochs', type=int, default=20)
        self.parser.add_argument('--last_epoch', type=int, default=0, help='>0 means continuing training')
        self.parser.add_argument('--lr', type=float, default=5e-4)
        self.parser.add_argument('--save_latest_freq', type=int, default= 225*30, help='save training model freq')
        self.parser.add_argument('--log_freq', type=int, default=225, help='log loss freq')
        self.parser.add_argument('--num_gpus', type=int, default=1)  # DDP strategy
        self.parser.add_argument('--ckpt_path', type=str,
                                 default='/root/zju/project/runs/nerf/fire2/ckpt_epoch17_iter7500_psnr-23.214c-25.351f.pkl',
                                 help='ckpt')

        # dataset
        self.parser.add_argument('--img_downscale', type=int, default=3,
                                 help='how much to downscale the images for Cambridge nerf')
        self.parser.add_argument('--data_root_dir', type=str, default='G:\\dataset\\Cambridge')
        self.parser.add_argument('--scene', type=str, default='StMarysChurch')
        self.parser.add_argument('--use_cache', type=bool, default=False, help='if use cache when loading rays')
        self.parser.add_argument('--if_save_cache', type=bool, default=True,
                                 help='if save cache to the data root dir when loading rays')

        # nerf related
        self.parser.add_argument('--layers', type=int, default=8, help='nerf MLPθ1 hidden layers')
        self.parser.add_argument('--W', type=int, default=256, help='nerf MLP1 hidden dim')
        self.parser.add_argument('--N_xyz_freq', type=int, default=15, help='position embedding freq numbers')
        self.parser.add_argument('--N_dir_freq', type=int, default=4, help='direction embedding freq numbers')
        self.parser.add_argument('--N_c', type=int, default=64, help='coarse samples')
        self.parser.add_argument('--N_f', type=int, default=128, help='fine samples')
        self.parser.add_argument('--use_disp', type=bool, default=False,
                                 help='sample xyz in disp(Inversely proportional to depth) space')
        self.parser.add_argument('--perturb', type=float, default=1.0,
                                 help='perturbation added to the coarse sampled xyz')

        # nerf-w related
        self.parser.add_argument('--encode_a', type=bool, default=True, help='if embed appearance ')
        self.parser.add_argument('--encode_t', type=bool, default=True, help='if embed transient')
        self.parser.add_argument('--a_dim', type=int, default=48, help='appearance embedding dim')
        self.parser.add_argument('--t_dim', type=int, default=16, help='transient embedding dim')
        self.parser.add_argument('--beta_min', type=float, default=0.03, help='minimum beta of transient noise')
        self.parser.add_argument('--lambda_u', type=float, default=0.1,
                                 help='coefficient of regular loss part of transient sigma')

    def into_opt(self, save_opt=True):
        opt = self.parser.parse_args()
        args = vars(opt)
        exp_dir = os.path.join(opt.root_dir, opt.exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
        if save_opt:
            with open(os.path.join(exp_dir, 'options.txt'), mode='w', encoding='utf-8') as f:
                f.write('--------------------Options--------------------\n')
                for k, v in sorted(args.items()):
                    f.write('%s: %s\n' % (str(k), str(v)))
                f.write('--------------------End--------------------\n')
            with open(os.path.join(exp_dir, 'opt.pkl'), mode='wb') as f:  # save namespace
                pickle.dump(opt, f)
        return opt
