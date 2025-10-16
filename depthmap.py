import os
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from kornia.geometry.conversions import rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix

class ARKITDataset(Dataset):
    def __init__(self, meta_json: str, selected_views_json: str, image_scale = 1.0, color = False):
        
        assert os.path.exists(meta_json)
        assert os.path.exists(selected_views_json)

        self.image_scale = image_scale
        self.color = color

        self.images = []
        self.masks = []
        self.depthmaps =  []
        self.intrinsics =  []
        self.transforms =  []

        self.read_meta_json(meta_json)

        self.groups = None
        self.read_selected_views(selected_views_json)

        # self.groups = [
        #     (1,2,3,4),
        #     (0,2,3,4),
        #     (0,1,3,4),
        #     (1,2,4,5),
        #     (2,3,5,6),
        #     (3,4,6,7),
        #     (3,4,5,7),
        #     (3,4,5,6)
        # ]

    def read_meta_json(self, meta_json: str):
        
        with open(meta_json) as f:
            frames = json.load(f)

        for frame in frames:
            image, mask, depthmap, K, transform = self.read_frame(frame)
        
            self.images.append(image)
            self.masks.append(mask)
            self.depthmaps.append(depthmap)
            self.intrinsics.append(K)
            self.transforms.append(transform)

    def read_selected_views(self, selected_view_json: str):
        with open(selected_view_json) as f:
            self.groups = json.load(f)

    def read_frame(self, frame: dict):

        image_scale = self.image_scale
        color = self.color

        assert os.path.exists(frame['image_name']), f'image {frame['image_name']} is not exists!'
        assert os.path.exists(frame['mask_name']), f'mask {frame['mask_name']} is not exists!'

        # ignore orientation
        image = cv2.imread(frame['image_name'], cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if color else cv2.COLOR_BGR2GRAY)

        mask = cv2.imread(frame['mask_name'], cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_GRAYSCALE)
        
        # (H,W,C) => (C,H,W)
        # image_tensor = torch.from_numpy(image).to(torch.float32)

        depthmap = np.fromfile(frame['depth_name'],np.float32)
        depthmap = depthmap.reshape(frame['depthmap_height'], frame['depthmap_width'])

        assert abs(depthmap.shape[0]/depthmap.shape[1] - image.shape[0]/image.shape[1]) < 1e-6

        intrinsic = np.loadtxt(frame['intrinic_name'], delimiter=',', dtype= np.float32)

        # convert to openmvs image coordinates
        K = intrinsic.reshape(3,3)
        K[0,2] -= 0.5
        K[1,2] -= 0.5

        if image_scale != 1.:
            image = cv2.resize(image, (0,0), fx = image_scale, fy = image_scale, interpolation = cv2.INTER_AREA)
            mask = cv2.resize(image, (0,0), fx = image_scale, fy = image_scale, interpolation = cv2.INTER_NEAREST)
            # scale intrinsic matrix by image scale
            K[0,0] *= image_scale
            K[0,2] = (K[0,2] + 0.5) * image_scale - 0.5
            K[1,1] *= image_scale
            K[0,2] = (K[1,2] + 0.5) * image_scale - 0.5            
        
        self.H, self.W = image.shape[:2]
        
        depthmap_scale = image.shape[0] / depthmap.shape[0]

        scaled_depthmap = depthmap

        # resize depthmap to image shape
        if depthmap_scale != 1.:
            depthmap[depthmap <=0] = 1e-7
            inv_depthmap = cv2.resize(1.0/ depthmap, (0,0), fx=depthmap_scale, fy=depthmap_scale, interpolation=cv2.INTER_LINEAR)
            scaled_depthmap = 1.0 / inv_depthmap


        transform = np.loadtxt(frame['extrinsic_name'], delimiter=',', dtype= np.float32)

        # convert to openmvs coordinates
        transform = self.convert_transform(transform.reshape(4,4))

        return image, mask, scaled_depthmap, K, transform

    def convert_transform(self, M: np.ndarray) -> np.ndarray:
        '''
            convert ARKIT extrinsic to openmvs coordinates and convert the camera2world extrinsic to world2camera extrinsic
        '''
        F = np.eye(4, dtype = np.float32)
        F[1,1] = F[2,2] = -1.

        # return rotation and camera center
        return F @ np.linalg.inv(M)
    
    def pointcloud(self, ply_file: str):
        from pointcloud import write_ply_optimized
        
        all_points = []
        all_colors = []

        y_coords, x_coords = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')

        # (H*W, 3) - 展平的 2D 齐次像素坐标 (x, y, 1)
        indexs_flat = np.stack([x_coords, y_coords, np.ones_like(y_coords)], axis=2).reshape(-1, 3).astype(np.float32)

        for index, image in enumerate(self.images):
            K_inv = np.linalg.inv(self.intrinsics[index]) # 3x3
            
            # M_inv 是 相机坐标系 -> 世界坐标系 的 4x4 变换矩阵 (T_wc)
            M_inv = np.linalg.inv(self.transforms[index]) 

            # 1. 反投影得到 3D 射线方向 D (在 z=1 平面上)
            # K_inv (3, 3) @ indexs_flat.T (3, H*W) => (3, H*W).T => (H*W, 3)
            directions = (K_inv @ indexs_flat.T).T 

            # 展平的深度图 (H*W, 1)
            depth_flat = self.depthmaps[index].reshape(-1, 1)

            # 2. 引入深度 Z，计算 3D 点 P_c (相机坐标系)
            # Z * D. P_c 形状: (H*W, 3)
            points_camera = depth_flat * directions 
            
            # 3. 转换为齐次坐标 (H*W, 4)
            # 在第 4 列添加 1
            points_cam_homo = np.concatenate([points_camera, np.ones((self.H * self.W, 1), dtype=np.float32)], axis=1)

            # 4. 转换到世界坐标 P_w
            # M_inv (4, 4) @ points_cam_homo.T (4, H*W) => (4, H*W).T => (H*W, 4)
            points_world_homo = (M_inv @ points_cam_homo.T).T

            # 5. 提取 P_w (世界坐标)，并恢复 (H, W, 3) 形状
            # 形状: (H, W, 3)
            points_world = points_world_homo[:, :3].reshape(self.H, self.W, 3)
            
            # 7. 存储展平后的 (H*W, 6) 数据
            all_points.append(points_world.reshape(-1, 3))
            all_colors.append(image.reshape(-1, 3))

        # 8. 循环结束后，合并所有点云块并写入文件
        if all_points:
            final_points = np.concatenate(all_points, axis=0)
            final_colors = np.concatenate(all_colors, axis=0)
            write_ply_optimized(ply_file, final_points, colors= final_colors)


    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        return group[0], group[1:]
    
class ARKIModule(nn.Module):
    def __init__(self, init_focal: float, W: int = 1920, H: int = 1440, dtype=torch.float32, device = 'cuda'):
        super().__init__()
        self.focal = nn.Parameter(torch.tensor(init_focal, dtype=dtype, device=device))
        self.H_WIN_SIZE = 7
        self.W = W
        self.H = H
        self.cx = W/2.
        self.cy = H/2.
        self._dtype = dtype
        self.device = device
        self.axis_angles = nn.ParameterList()
        self.centers = nn.ParameterList()

        # 创建像素坐标网格
        self.y_coords, self.x_coords = torch.meshgrid(
            torch.arange(self.H, device=self.device, dtype=torch.float32),
            torch.arange(self.W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )

    def initialize_extrinsics(self, arkit_extrinsics: list):
        for extrinsic in arkit_extrinsics:
            R =  extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            C = -R.T @ t

            _T = lambda x: torch.from_numpy(x).clone().to(dtype=self._dtype, device=self.device)

            # convert rotation matrix to angle-axis
            rotvec = rotation_matrix_to_axis_angle(_T(R))

            self.axis_angles.append(nn.Parameter(rotvec))
            self.centers.append(nn.Parameter(_T(C)))

    def compute_image_normals_cpu(self, depthmap: torch.Tensor):
        normals = torch.zeros(*depthmap.shape, 3).to(dtype=depthmap.dtype, device= depthmap.device)

        for i in range(depthmap.shape[0]):
            for j in range(depthmap.shape[1]):
                normal = self.compute_normal_cpu((j,i), depthmap)
                
                if normal is not None:
                    normals[i,j] = normal[0]

        return normals

    def compute_normal_cpu(self, pos: tuple, depthmap: torch.Tensor):
        assert depthmap.shape[:2] == (self.H, self.W)

        x,y = pos

        assert 0<= x < self.W and 0<= y < self.H

        if x<=0 or x >= self.W -1 or y <=0 or y >= self.H -1:
            return None

        depth = depthmap[y,x]

        if depth.item() < 1e-6:
            return None
        
        depth_left   = depthmap[y, x - 1].item()
        depth_right  = depthmap[y, x + 1].item()
        depth_top    = depthmap[y - 1, x].item()
        depth_bottom = depthmap[y + 1, x].item()
        
        if depth_left < 1e-6 or depth_right < 1e-6 or depth_top < 1e-6 or depth_bottom < 1e-6:
            return None

        d_u = 0.5 * (depth_right - depth_left)
        d_v = 0.5 * (depth_bottom - depth_top)

        # compute normal from depth gradient
        normal = torch.stack([ self.focal * d_u, self.focal * d_v, (self.cx- x) * d_u + (self.cy- y) * d_v - depth ]).to(self.device)

        norm_value = torch.norm(normal)

        return None if norm_value < 1e-6 else (normal / norm_value, depth)

    def compute_normal_cuda(self, depthmap: torch.Tensor):
        # (1 * 1 * W * H)
        depth4d = depthmap[None, None, :, :]

        # X方向梯度
        kernel_x = torch.tensor([[-0.5, 0, 0.5]], device=self.device).view(1, 1, 1, 3)
        d_u = F.conv2d(depth4d, kernel_x, padding=(0, 1)).squeeze()

        # Y方向梯度  
        kernel_y = torch.tensor([[-0.5], [0], [0.5]], device=self.device).view(1, 1, 3, 1)
        d_v = F.conv2d(depth4d, kernel_y, padding=(1, 0)).squeeze()

        # 计算法向量 (基于平面假设)
        normal_x = self.focal * d_u
        normal_y = self.focal * d_v
        normal_z = (self.cx - self.x_coords) * d_u + (self.cy - self.y_coords) * d_v - depthmap

        normal_map = torch.stack([normal_x, normal_y, normal_z], dim=-1)  # (H, W, 3)

        norm = torch.norm(normal_map, dim=-1, keepdim=True)

        return normal_map / (norm + 1e-8)

    def compute_H_matrix_batch(self, ref_camera: int, ref_depthmap: torch.Tensor, target_camera: int):
        '''
        The homography matrix for single pixel,
        H =K_2 R_2(R_1^T + (C_1 - C_2)n_d^T) K_1^{-1}
        '''

        T = lambda x: torch.tensor(x, device = self.device, dtype = self._dtype)
        f = self.focal

        # self.focal is the only trainable parameter
        K = torch.stack([
            torch.stack([f ,    T(.0),  T(self.cx - 0.5)]),
            torch.stack([T(.0), f ,     T(self.cy - 0.5)]),
            torch.stack([T(.0), T(.0),  T(1.)])
        ]) 

        K_inv = torch.inverse(K)

        # (3, 3)
        R1: torch.Tensor = axis_angle_to_rotation_matrix(self.axis_angles[ref_camera][None]).squeeze()
        # (3,)
        C1: torch.Tensor = self.centers[ref_camera]    
        # (3, 3)
        R2: torch.Tensor = axis_angle_to_rotation_matrix(self.axis_angles[target_camera][None]).squeeze()
        # (3,)
        C2: torch.Tensor = self.centers[target_camera]    
        
        # (3, 1)
        C_diff = (C1 - C2).contiguous().view(3, 1)
        
        # (H, W, 3)
        normal = self.compute_normal_cuda(ref_depthmap)  

        # (H, W, 3)
        n_d = normal / (ref_depthmap[..., None] + 1e-8) 

        # Auto broadcast: (3, 1) * (H, W, 1, 3) => (H, W, 3, 3)
        inner = C_diff @ n_d.view(*n_d.shape[:2] ,1, 3)

        # (3, 3) + (H, W, 3, 3) =>(H, W, 3, 3)
        H_2  = R1.T + inner

        # [(3, 3) * (3, 3)] * (H, W, 3, 3) * (3, 3) => (3, 3) * (H, W, 3, 3) * (3, 3) => (H, W, 3, 3) * (3, 3) => (H, W, 3, 3)
        # - `(K @ R2) @ H_2 @ K_inv` browadcast only one time
        # - `K @ (R2@ H_2) @ K_inv` will broadcast two times
        H = (K @ R2) @ H_2 @ K_inv

        # (H, W, 3, 3)
        return H

    # H (H, W, 3, 3)
    def homography_warp_batch(self, target_image: torch.Tensor, H_matrix: torch.Tensor):
        assert target_image.dim() == 2

        H, W = target_image.shape
        
        # (H, W, 3)
        coords = torch.stack([self.x_coords, self.y_coords, torch.ones_like(self.y_coords)], dim=-1)  # (H, W, 3)

        # (H, W, 3, 3) * (H, W, 3, 1) => (H, W, 3, 1) => (H, W, 3)
        warped_coords = (H_matrix @ coords[..., None]).squeeze()

        # (H, W, 2)
        warped_xy = warped_coords[...,:2] / (warped_coords[..., 2:3] + 1e-8)
        
        # (H, W, 2)
        # normalize the x and y to [-1,1] by W and H
        warped_xy_norm = 2 * warped_xy / torch.tensor([W -1, H - 1], dtype= warped_xy.dtype, device= warped_xy.device) - 1.

        # sample RGB by warpped coordinates
        warped_image = F.grid_sample(target_image[None, None, ...], warped_xy_norm[None, ...], align_corners=True, mode='bilinear', padding_mode='zeros')
        
        # (H, W)
        return warped_image.squeeze()

    def compute_ZNCC_fast(self, ref_gray: torch.Tensor, tgt_gray: torch.Tensor, window_size: int = 7):
        assert ref_gray.dim() == tgt_gray.dim() and ref_gray.dim() == 2
        
        # (1, 1, window_size, window_size)
        kernel = torch.ones(1, 1, window_size, window_size, device=self.device) / (window_size * window_size)
        
        # 局部均值
        # (H, W)
        ref_mean = F.conv2d(ref_gray[None, None, ...], kernel, padding=window_size//2).squeeze()
        tgt_mean = F.conv2d(tgt_gray[None, None, ...], kernel, padding=window_size//2).squeeze()
        
        # 零均值化 (H, W)
        ref_centered = ref_gray - ref_mean
        tgt_centered = tgt_gray - tgt_mean
        
        # 协方差 (H, W)
        cov = ref_centered * tgt_centered
        cov_sum = F.conv2d(cov[None, None, ...], kernel, padding=window_size//2).squeeze()

        # 方差 (H, W)
        ref_var = ref_centered ** 2
        tgt_var = tgt_centered ** 2

        ref_var_sum = F.conv2d(ref_var[None, None, ...], kernel, padding=window_size//2).squeeze()
        tgt_var_sum = F.conv2d(tgt_var[None, None, ...], kernel, padding=window_size//2).squeeze()
        
        # ZNCC
        denominator = torch.sqrt(ref_var_sum * tgt_var_sum + 1e-8)
        zncc_map = cov_sum / denominator
        
        # 输出: (H, W)
        return zncc_map

    def compute_ZNCC_batch(self, ref_camera: int, ref_image: torch.Tensor, ref_depthmap: torch.Tensor, 
                      target_camera: int, target_image: torch.Tensor,
                      ref_mask: torch.Tensor = None, target_mask: torch.Tensor = None):
        
        #homograph matrix
        H_batch = self.compute_H_matrix_batch(ref_camera, ref_depthmap, target_camera)

        # warp target image to ref coordinates
        warped_target = self.homography_warp_batch(target_image, H_batch)  

        # bound mask
        valid_mask = (warped_target != 0)

        # mask by target image mask
        valid_mask = valid_mask & target_mask if target_mask is not None else valid_mask
        
        # mask by ref image mask
        if ref_mask is not None:
            warped_ref_mask_to_target = self.homography_warp_batch(ref_mask, H_batch)

            # 得到目标图有效区域
            valid_mask_target = warped_ref_mask_to_target > 0.5
            valid_mask = valid_mask & valid_mask_target

        valid_count = valid_mask.sum()

        if valid_count == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0, device=self.device)

        # (H, W)
        zncc_map = self.compute_ZNCC_fast(ref_image, warped_target)  
        
        zncc_valid = zncc_map * valid_mask

        return zncc_valid.sum(), valid_count
    
    def warp_image(self, ref_camera: int, ref_image: torch.Tensor, ref_depthmap: torch.Tensor, 
                target_cameras: list, target_images: dict, target_depthmaps:list,
                ref_mask: torch.Tensor = None, target_masks: list = None):
        total_score = 0.
        total_pixels = 0.

        for target_camera in target_cameras:
            
            target_mask = None if target_masks is None else target_masks[target_camera]

            zncc_map, n_pixels = self.compute_ZNCC_batch(
                ref_camera, ref_image, ref_depthmap,
                target_camera, target_images[target_camera],
                ref_mask, target_mask
            )

            if zncc_map is not None and n_pixels > 0:
                total_score = total_score + zncc_map
                total_pixels += n_pixels

        if total_pixels == 0:
            # no valid pixels, return zero loss
            return torch.tensor(0., device=self.device, dtype=self.focal.dtype)
        
        # ZNCC range [-1,1], loss = 1 - mean ZNCC
        mean_zncc = total_score / total_pixels
        loss = 0.5 * (1. - mean_zncc)
        return loss
    
    def reproject_image(self, ref_camera:    int, ref_image:    torch.Tensor, ref_depthmap:    torch.Tensor,    ref_mask:    torch.Tensor,
                              target_camera: int, target_image: torch.Tensor, target_depthmap: torch.Tensor,    target_mask: torch.Tensor):
        
        T = lambda x: torch.tensor(x, device = self.device, dtype = self._dtype)
        f = self.focal

        # self.focal is the only trainable parameter
        K = torch.stack([
            torch.stack([f ,    T(.0),  T(self.cx - 0.5)]),
            torch.stack([T(.0), f ,     T(self.cy - 0.5)]),
            torch.stack([T(.0), T(.0),  T(1.)])
        ]) 

        last_row = torch.tensor([[0, 0, 0, 1]], dtype=self._dtype, device=self.device)

        # (3, 3)
        R1: torch.Tensor = axis_angle_to_rotation_matrix(self.axis_angles[ref_camera][None]).squeeze()
        # (3,)
        C1: torch.Tensor = self.centers[ref_camera]
        t1: torch.Tensor = -R1 @ C1

        M1 = torch.cat([R1, t1.view(3, 1)], dim=1) 
        # (4, 4)
        ref_transform = torch.cat([M1, last_row], dim=0)  

        # (3, 3)
        R2: torch.Tensor = axis_angle_to_rotation_matrix(self.axis_angles[target_camera][None]).squeeze()
        # (3,)
        C2: torch.Tensor = self.centers[target_camera]
        t2: torch.Tensor = -R2 @ C2
        # (3, 4)
        M2 = torch.cat([R2, t2.view(3, 1)], dim=1) 
        # (4, 4)
        target_transform = torch.cat([M2, last_row], dim=0)  

        x_coords, y_coords = self.x_coords, self.y_coords
        # (H, W, 3)
        coords = torch.stack([x_coords, y_coords, torch.ones_like(y_coords)], dim=-1)  

        # (H, W, 3)
        camera_XYZ = (K.inverse() @ coords[...,None]).squeeze() * ref_depthmap[...,None]
        ones = torch.ones((*camera_XYZ.shape[:2], 1), device = camera_XYZ.device, dtype = camera_XYZ.dtype)
        # (H, W, 4)
        h_camera_XYZ = torch.cat([camera_XYZ, ones], dim=-1)

        # (4, 4) * (H, W, 4, 1) => (H, W, 4, 1)
        world_XYZ = (ref_transform.inverse() @ h_camera_XYZ[..., None])

        # (4, 4) & (H, W, 4, 1) => (H, W, 4, 1) => (H, W, 4)
        h_target_camera_XYZ = (target_transform @ world_XYZ).squeeze(-1)
        # (H, W, 3)
        target_camera_XYZ = h_target_camera_XYZ[:,:, :3]
        # (3, 3) * (H, W, 3, 1) => (H, W, 3, 1) => (H, W, 3)
        h_target_camera_uv = (K @target_camera_XYZ[..., None]).squeeze(-1)
        # (H, W)
        target_camera_uv = h_target_camera_uv[...,:2]/(h_target_camera_uv[...,2:3] + 1e-8)

        scale = torch.tensor([self.W -1, self.H -1],dtype=torch.float32, device=self.device)

        # (H, W, 2)
        target_camera_uv = 2 * target_camera_uv/scale.view(1, 1, 2) -1.

        projected_image = F.grid_sample(target_image[None, None, ...], target_camera_uv[None, ...], align_corners=True, mode='bilinear', padding_mode='zeros')
        # (H, W)
        projected_image = projected_image.squeeze()

        valid_mask = (projected_image > 1e-3) & (ref_image > 1e-3)

        if target_mask is not None:
            projected_mask = F.grid_sample(target_mask[None, None, ...].float(), target_camera_uv[None, ...], align_corners=True, mode='nearest', padding_mode='zeros')
            projected_mask = projected_mask.squeeze()
            valid_mask = valid_mask & (projected_mask > 0)

        if ref_mask is not None:
            valid_mask = valid_mask & (ref_mask > 0)

        if valid_mask.sum() > 0:
            error = torch.abs(ref_image - projected_image)[valid_mask].mean()
        else:
            error = torch.tensor(0.0, device=self.device)

        return error * 255

    def forward(self, ref_camera:     int,  ref_image:      torch.Tensor, ref_depthmap:     torch.Tensor, ref_mask:     torch.Tensor,
                      target_cameras: list, target_images:  dict,         target_depthmaps: list,         target_masks: list):
        losses = []

        # reproject
        for target_camera in target_cameras: 
            target_image = target_images[target_camera]
            target_depthmap = target_depthmaps[target_camera]
            target_mask = target_masks[target_camera]

            loss = self.reproject_image(ref_camera, ref_image, ref_depthmap, ref_mask,
                                        target_camera, target_image, target_depthmap, target_mask)
            losses.append(loss)

        return sum(losses)/len(target_cameras)
    
class ARKITModel(object):
    
    def __init__(self, meta_json: str, selected_views_json:str, device='cuda'):
        # self.dataset = ARKITDataset(meta_json, image_scale= 256/1920.)
        self.dataset = ARKITDataset(meta_json, selected_views_json)
        H, W = self.dataset.images[0].shape[:2]

        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.device = device
        self.arkit_module = ARKIModule(1418.71,H=H, W=W, device=device)
        self.arkit_module.initialize_extrinsics(self.dataset.transforms)

        self.image_tensros = [torch.from_numpy(image).to(device = self.device, dtype= torch.float32) for image in self.dataset.images]
        self.mask_tensros = [torch.from_numpy(mask).to(device = self.device, dtype= torch.float32) for mask in self.dataset.masks]
        self.depthmap_tensors = [torch.from_numpy(depthmap).to(self.device) for depthmap in self.dataset.depthmaps]

    def test_view_angle_diff(self, ref_camera: int):
        # ref_image = self.image_tensros[ref_camera]
        depthmap = self.depthmap_tensors[ref_camera]
        # (H, W, 3)
        cuda_normals = self.arkit_module.compute_normal_cuda(depthmap)
        # cpu_normals = self.arkit_module.compute_image_normals_cpu(depthmap)

        # view directon
        K  = torch.tensor([[self.arkit_module.focal, 0, self.arkit_module.cx],
                           [0, self.arkit_module.focal, self.arkit_module.cy],
                           [0,0,1]],dtype=torch.float32, device= depthmap.device)
        K_inv = K.inverse()

        x_coords, y_coords = self.arkit_module.x_coords, self.arkit_module.y_coords

        # (H, W, 3)
        coords = torch.stack([x_coords, y_coords, torch.ones_like(y_coords)], dim=-1)  

        # (H, W, 3)
        XYZ = (K_inv @ coords[...,None]).squeeze() * depthmap[...,None]

        # angles
        angles = (XYZ *cuda_normals).sum(dim= -1)

        # return torch.square(cuda_normals - cpu_normals).mean().sqrt(), angles
        return 0, (angles < 0).sum()/ (depthmap.shape.numel())

    def test_warping(self, ref_camera: int, target_camera: int, save_diff: bool = True):
        ref_image = self.image_tensros[ref_camera]/255.

        target_image = self.image_tensros[target_camera]/255.

        depthmap = self.depthmap_tensors[ref_camera]
        
        H_matrix = self.arkit_module.compute_H_matrix_batch(ref_camera, depthmap, target_camera)
        warped_image = self.arkit_module.homography_warp_batch(target_image, H_matrix)

        x_coords, y_coords = self.arkit_module.x_coords, self.arkit_module.y_coords
        # (H, W, 3)
        coords = torch.stack([x_coords, y_coords, torch.ones_like(y_coords)], dim=-1)  

        # view directon
        K  = torch.tensor([[self.arkit_module.focal, 0, self.arkit_module.cx],
                           [0, self.arkit_module.focal, self.arkit_module.cy],
                           [0,0,1]],dtype=torch.float32, device= depthmap.device)

        # (H, W, 3)
        camera_XYZ = (K.inverse() @ coords[...,None]).squeeze() * depthmap[...,None]
        ones = torch.ones((*camera_XYZ.shape[:2], 1), device = camera_XYZ.device, dtype = camera_XYZ.dtype)
        # (H, W, 4)
        h_camera_XYZ = torch.cat([camera_XYZ, ones], dim=-1)

        ref_transform = self.dataset.transforms[ref_camera]
        ref_transform = torch.from_numpy(ref_transform).to(device= self.device)

        # (4, 4) * (H, W, 4, 1) => (H, W, 4, 1)
        world_XYZ = (ref_transform.inverse() @ h_camera_XYZ[..., None])

        target_transform = self.dataset.transforms[target_camera]
        target_transform = torch.from_numpy(target_transform).to(device= self.device)

        # (4, 4) & (H, W, 4, 1) => (H, W, 4, 1) => (H, W, 4)
        h_target_camera_XYZ = (target_transform @ world_XYZ).squeeze(-1)
        # (H, W, 3)
        target_camera_XYZ = h_target_camera_XYZ[:,:, :3]
        # (3, 3) * (H, W, 3, 1) => (H, W, 3, 1) => (H, W, 3)
        h_target_camera_uv = (K @target_camera_XYZ[..., None]).squeeze(-1)
        # (H, W)
        target_camera_uv = h_target_camera_uv[...,:2]/(h_target_camera_uv[...,2:3] + 1e-8)

        scale = torch.tensor([self.arkit_module.W -1, self.arkit_module.H -1],dtype=torch.float32, device=self.device)

        # (H, W, 2)
        target_camera_uv = 2 * target_camera_uv/scale.view(1, 1, 2) -1.

        projected_image = F.grid_sample(target_image[None, None, ...], target_camera_uv[None, ...], align_corners=True, mode='bilinear', padding_mode='zeros')
        # (H, W)
        projected_image = projected_image.squeeze()

        valid_mask = (projected_image > 1e-3) & (ref_image > 1e-3)
        
        if valid_mask.sum() > 0:
            error_map = 255 * torch.abs(ref_image - projected_image)
            reproject_error = error_map[valid_mask].mean()
            if save_diff:
                torchvision.utils.save_image(error_map, f'diff_{ref_camera}_{target_camera}.png')
        else:
            reproject_error = torch.tensor(0.0, device=self.device)

        return (warped_image>0).sum(), (projected_image > 0).sum(), reproject_error

    def train(self, epochs=100, lr=2e-4):
        self.arkit_module.train()
        # optimizer = torch.optim.Adam(self.arkit_module.parameters(), lr=lr, eps=1e-08, betas=(0.5, 0.9))
        optimizer = torch.optim.Adam(
            self.arkit_module.parameters(),
            lr=1e-4,                    # 较小的学习率，姿态需要精细调整
            betas=(0.9, 0.99),          # 较小的第二动量，避免震荡
            eps=1e-8,
            weight_decay=1e-5           # 较小的权重衰减
        )
        MAX_TARGET_CAMMERS = 5

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            for ref_camera, target_cameras in self.data_loader:
                target_cameras = target_cameras[:MAX_TARGET_CAMMERS]

                # normalize image to [0,1]
                ref_image = self.image_tensros[ref_camera]/255.
                ref_depthmap = self.depthmap_tensors[ref_camera]
                ref_mask = self.mask_tensros[ref_camera]

                target_images = {i: self.image_tensros[i]/255. for i in target_cameras}
                target_masks = {i: self.mask_tensros[i] for i in target_cameras}
                target_depthmaps = {i: self.depthmap_tensors[i] for i in target_cameras}

                loss = self.arkit_module(ref_camera, ref_image, ref_depthmap, ref_mask,
                                         target_cameras, target_images, target_depthmaps, target_masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                
                # print(f"[Epoch {epoch+1}/{epochs}] Ref Camera: {ref_camera} | Loss: {loss.item():.5f}")

            ref_rotation = axis_angle_to_rotation_matrix(self.arkit_module.axis_angles[0][None]).squeeze()
            print(f'Ref image {0} rotation: {ref_rotation}, focal: {self.arkit_module.focal.item()}')

            mean_loss = total_loss / max(1, n_batches)
            print(f"==== Epoch {epoch+1}/{epochs} done | mean loss = {mean_loss:.6f} ====")

if __name__ =='__main__':
    meta_path = '/data/reconstruction/costa/arkit/arkit_meta.json'
    selected_views_path = '/data/reconstruction/costa/arkit_mvs/selected_views.json'

    model = ARKITModel(meta_path, selected_views_path)
    model.train()

    # diff = model.test_view_angle_diff(0)
    # print(f'view angle diff: {diff}')

    # ref_camera = 0

    # with torch.no_grad():
    #     for target_camera in range(1,4):
    #         diff = model.test_warping(ref_camera, target_camera, save_diff=False)
    #         print(f'Ref camera: {ref_camera}, target camera: {target_camera}, warped valid: {diff[0]}, projected valid: {diff[1]}, reproject error: {diff[2]}')
    
    # dataset = ARKITDataset('/data/reconstruction/clock/arkit/arkit_meta.json', image_scale=0.5,color=True)
    # dataset.pointcloud('/data/reconstruction/clock/python_ply.ply')
