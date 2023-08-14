'''
生成合成视图
Input: 1. scene点云 2. train set view 3. bounding box
Output: 1. novel_views.ply(可视化) 2. novel_views.pkl 数据包括:[img_size, Ks, c2ws, nears, fars,nearest_ids]
'''
import os.path
import pickle
from scipy.spatial import KDTree

from dataset.cambridge import CambridgeDataset
from dataset.seven_scenes import SevenScenesDataset
from utils.utils import *

# cambridge: d_scene=2m, d_view=6m, theta=9°
d_scene = 0.15  # 离scene要远一点，否则会有点在scene内部。均与原场景尺度一致(scale前)
d_view = 0.15  # 离view要近一点, 保证novel view合法、一致
theta = 9 / 180 * np.pi  # 四元数旋转扰动幅度, 每个轴[-θ/2,θ/2] 若pose表示c2w,添加扰动后变为c'2w可视为c'->c,c->w两个变换, 右扰动

n_novel = 4000  # 要求novel views > n_novel
start_resolution = 2
delta_r = 1
if __name__ == '__main__':
    opt_path = './runs/nerf/fire/opt.pkl'
    with open(opt_path, mode='rb') as f:
        opt = pickle.load(f)
    # opt.data_root_dir = 'E:\\nerf\\Cambridge'
    # opt.scene = 'StMarysChurch'
    valid_dataset = SevenScenesDataset(opt.data_root_dir, opt.scene, split='valid', img_downscale=opt.img_downscale,
                                       use_cache=True)
    # 保持参数尺度一致
    d_scene, d_view = d_scene / valid_dataset.scale_factor, d_view / valid_dataset.scale_factor
    print('scale factor:%f' % valid_dataset.scale_factor)
    # 1. load scene and bounding box
    cache_dir = os.path.join(opt.data_root_dir, opt.scene, 'cache')
    scene = np.load(os.path.join(cache_dir, 'box_scene.npy'))
    bounding_box = np.load(os.path.join(cache_dir, 'bounding_box.npy'))
    print('bounding box:\n', bounding_box)
    scene_kdtree = KDTree(scene[:, :3])  # novel view候选box内场景KDTree
    # 2. load train views
    centers = [valid_dataset.view_c2w[id][:, -1] for id in valid_dataset.train_set]
    quats = [rotmat2quat(valid_dataset.view_c2w[id][:, :3]) for id in valid_dataset.train_set]
    centers = np.vstack(centers)
    quats = np.vstack(quats)
    train_view_kdtree = KDTree(centers)  # train set camera position KDTree

    # 3. meshgrid and synthesis novel view
    print('to synthesis %d novel view...' % n_novel)
    for r in range(start_resolution, 4096, delta_r):
        d_i = np.min(bounding_box[:, 1] - bounding_box[:, 0]) / r
        candidates = np.mgrid[bounding_box[0, 0]:bounding_box[0, 1]:d_i, bounding_box[1, 0]:bounding_box[1, 1]:d_i,
                     bounding_box[2, 0]:bounding_box[2, 1]:d_i]
        candidates = candidates.reshape(3, -1).T
        mesh_num = len(candidates)
        # ①删除距离scene小于d_scene的候选点; ②删除与最近train view距离大于d_view的候选position
        dist, ind = scene_kdtree.query(candidates)
        mask1 = dist > d_scene
        dist, ind = train_view_kdtree.query(candidates)
        mask2 = dist < d_view
        mask = mask1 & mask2
        candidates = candidates[mask]
        dist = dist[mask]
        ind = ind[mask]
        print('[resolution:%d] [candidates:%d, reserved candidates:%d]' % (r, mesh_num, len(dist)))
        if len(dist) >= n_novel:
            print('completed, finally synthesis %d novel views' % len(dist))
            n_novel = len(dist)
            break
    # 4. 添加扰动c2w -> c'2w, 当然optical center不变
    # 单位四元数在四元数乘法意义上是群, 单位元e:(w,x,y,z)=(1,0,0,0),即旋转角=0。逆元为共轭
    novel_centers = candidates  # novel view optical center
    old_quats = quats[ind]  # nearest train view quats, c2w
    perturb_angle_xyz = (np.random.rand(len(novel_centers), 3) - 0.5) * theta
    pertub_quant = np.hstack(  # 扰动四元数,c2w, 右扰法c'2w即R(c->w)×R(c'->c)
        [np.cos(np.linalg.norm(perturb_angle_xyz, axis=1, keepdims=True)), np.sin(perturb_angle_xyz)])
    novel_quats = quat_mult(old_quats, pertub_quant)
    novel_quats = novel_quats / np.linalg.norm(novel_quats, axis=1, keepdims=True)  # 归一化
    novel_c2w = [np.hstack([quat2rotmat(quat), novel_centers[i].reshape(3, 1)]) for i, quat in enumerate(novel_quats)]

    # 5. 将novel view相关data保存至cache内
    with open(os.path.join(opt.data_root_dir, opt.scene, 'cache', 'novel_views.pkl'), mode='wb') as f:
        novel_nearest_id = [valid_dataset.train_set[ind[i]] for i in range(n_novel)]
        # novel_K = [valid_dataset.view_K[valid_dataset.train_set[ind[i]]] for i in range(n_novel)]
        novel_K = [valid_dataset.K for i in range(n_novel)]  # 7 scenes fixed K
        novel_near = [valid_dataset.view_near[valid_dataset.train_set[ind[i]]] for i in range(n_novel)]
        novel_far = [valid_dataset.view_far[valid_dataset.train_set[ind[i]]] for i in range(n_novel)]
        data = [valid_dataset.downscale_size, novel_K, novel_c2w, novel_near, novel_far,
                novel_nearest_id]  # img_size, K, c2w
        pickle.dump(data, f)
        print('novel view data saved to cache')

    # 6. 生成novel view可视化的ply文件
    cam_pts_idx = dict()
    cam_rgb = dict()
    camera_edge = 0.05
    points = np.array([[0, 0, 0]], dtype=float)
    colors = np.array([[0, 0, 0]], dtype=np.uint8)
    # 每个camera生成5个points
    for i in range(len(novel_c2w)):
        c2w = novel_c2w[i]
        # K = valid_dataset.view_K[valid_dataset.train_set[ind[i]]]  # 取最近train camera的K
        K = valid_dataset.K
        img_w, img_h = valid_dataset.downscale_size[0], valid_dataset.downscale_size[1]
        ray_o, ray_d = get_corner_ray(img_w, img_h, K, c2w)
        corner = ray_o.reshape(1, 3) + ray_d * camera_edge
        num = len(points)
        points = np.vstack([points, ray_o.reshape(1, 3), corner])
        # train-> red, test-> blue; NOVEL VIEW => GREEN
        color = np.array([[0, 255, 0]])
        colors = np.vstack([colors, np.tile(color, (5, 1))])
        cam_pts_idx[i] = list(range(num, num + 5))
        cam_rgb[i] = color[0]
    # save novel camera view ply
    path = os.path.join(opt.data_root_dir, opt.scene, 'novel_views.ply')
    with open(path, 'w') as f:
        f.write(ply_header.format(len(points), n_novel, n_novel * 4))
    with open(path, 'a') as f:
        # write points
        np.savetxt(f, np.c_[np.hstack([points, colors])], fmt='%.6f %.6f %.6f %d %d %d')
        # write cameras faces
        opacity = 16
        for id in range(n_novel):
            x = cam_pts_idx[id]
            r, g, b = cam_rgb[id][0], cam_rgb[id][1], cam_rgb[id][2]
            f.write('4 {} {} {} {} {} {} {} {}\n'.format(*x[1:], r, g, b, opacity))
        # write cameras edges
        for id in range(n_novel):
            x = cam_pts_idx[id]
            r, g, b = cam_rgb[id][0], cam_rgb[id][1], cam_rgb[id][2]
            f.write('{} {} {} {} {}\n'.format(x[0], x[1], r, g, b))
            f.write('{} {} {} {} {}\n'.format(x[0], x[2], r, g, b))
            f.write('{} {} {} {} {}\n'.format(x[0], x[3], r, g, b))
            f.write('{} {} {} {} {}\n'.format(x[0], x[4], r, g, b))
