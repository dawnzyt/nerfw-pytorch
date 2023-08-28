'''
构造可视化所有train、test camera view 的ply文件。
'''
import os.path
import pickle
from dataset.cambridge import CambridgeDataset
from dataset.seven_scenes import SevenScenesDataset
from utils.utils import *
camera_edge = 0.05

if __name__ == '__main__':
    opt_path = './runs/nerf/fire/opt.pkl'
    with open(opt_path, mode='rb') as f:
        opt = pickle.load(f)
    # opt.data_root_dir = 'E:\\nerf\\Cambridge'
    # opt.scene = 'StMarysChurch'
    valid_dataset = SevenScenesDataset(opt.data_root_dir, opt.scene, split='valid', img_downscale=opt.img_downscale,
                                     use_cache=True)

    ###################################
    # 1. save sfm points, 可注释       #
    # points, colors = valid_dataset.points[:, :3], (valid_dataset.points[:, 3:]).astype(np.uint8)
    # p_norm = np.linalg.norm(points, axis=1)
    # mx_norm = np.percentile(p_norm, 99.9)
    # points = points[p_norm < mx_norm]
    # colors = colors[p_norm < mx_norm]
    # print('points shape: ', points.shape)
    # # save sfm points ply
    # path = os.path.join(opt.data_root_dir, opt.scene, 'sfm_points.ply')
    # with open(path, 'w') as f:
    #     f.write(ply_header.format(len(points), 0, 0))
    # with open(path, 'a') as f:
    #     np.savetxt(f, np.c_[np.hstack([points, colors])], fmt='%.6f %.6f %.6f %d %d %d')
    ###################################
    #       2. 构造camera views        #
    cam_pts_idx = dict()
    cam_rgb = dict()
    n_cameras = valid_dataset.N_views
    points = np.zeros((1, 3), dtype=float)
    colors = np.zeros((1, 3), dtype=np.uint8)
    for id in valid_dataset.train_set + valid_dataset.test_set:
        c2w = valid_dataset.view_c2w[id]
        # K=valid_dataset.view_K[id]
        K = valid_dataset.K # 7 scenes fixed K
        img_w, img_h = valid_dataset.downscale_size[0], valid_dataset.downscale_size[1]
        ray_o, ray_d = get_corner_ray(img_w, img_h, K, c2w)
        corner = ray_o.reshape(1, 3) + ray_d * camera_edge
        num = len(points)
        points = np.vstack([points, ray_o.reshape(1, 3), corner])
        # train-> red, test-> blue
        color = np.array([[255, 0, 0]] if id in valid_dataset.train_set else [[0, 0, 255]])
        colors = np.vstack([colors, np.tile(color, (5, 1))])
        cam_pts_idx[id] = list(range(num, num + 5))
        cam_rgb[id] = color[0]
    # save camera views ply
    path = os.path.join(opt.data_root_dir, opt.scene, 'camera_views.ply')
    with open(path, 'w') as f:
        f.write(ply_header.format(len(points), n_cameras, n_cameras * 4))
    with open(path, 'a') as f:
        # write points
        np.savetxt(f, np.c_[np.hstack([points, colors])], fmt='%.6f %.6f %.6f %d %d %d')
        # write cameras faces
        opacity = 16
        for id in range(n_cameras):
            x = cam_pts_idx[id]
            r, g, b = cam_rgb[id][0], cam_rgb[id][1], cam_rgb[id][2]
            f.write('4 {} {} {} {} {} {} {} {}\n'.format(*x[1:], r, g, b, opacity))
        # write cameras edges
        for id in range(n_cameras):
            x = cam_pts_idx[id]
            r, g, b = cam_rgb[id][0], cam_rgb[id][1], cam_rgb[id][2]
            f.write('{} {} {} {} {}\n'.format(x[0], x[1], r, g, b))
            f.write('{} {} {} {} {}\n'.format(x[0], x[2], r, g, b))
            f.write('{} {} {} {} {}\n'.format(x[0], x[3], r, g, b))
            f.write('{} {} {} {} {}\n'.format(x[0], x[4], r, g, b))
    print('camera views saved to: ', path)
