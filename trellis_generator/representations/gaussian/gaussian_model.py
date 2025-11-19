import torch
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from .general_utils import inverse_sigmoid, strip_symmetric, build_scaled_rotation_matrices
from .sh_utils import SHRotator


class Gaussian:
    def __init__(
            self, 
            aabb : list,
            sh_degree : int = 0,
            mininum_kernel_size : float = 0.0,
            scaling_bias : float = 0.01,
            opacity_bias : float = 0.1,
            scaling_activation : str = "exp",
            device='cuda'
        ):
        self.init_params = {
            'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size 
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaled_rotation_matrices(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
        self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias)).cuda()
        self.rots_bias = torch.zeros((4)).cuda()
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).cuda()

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size ** 2
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=2) if self._features_rest is not None else self._features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])
    
    def from_scaling(self, scales):
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias
        
    def from_rotation(self, rots):
        self._rotation = rots - self.rots_bias[None, :]
    
    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        
    def from_features(self, features):
        self._features_dc = features
        
    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        # convert to actual gaussian attributes
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        if self.sh_degree > 0:
            features_extra = torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device=self.device))
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)
        
        # convert to _hidden attributes
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        if self.sh_degree > 0:
            self._features_rest = features_extra
        else:
            self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        self._scaling = self.inverse_scaling_activation(torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)) - self.scale_bias
        self._rotation = rots - self.rots_bias[None, :]

    def rotate_by_euler_angles(self, x: float, y: float, z: float):
        """
        rotate in z-y-x order, radians as unit
        """

        return Rotation.from_euler('xyz', [x, y, z], degrees=True).as_matrix().astype(np.float32)

    def transform_xyz(self, T, R, S, xyz):
        world_transformed = (xyz @ (R @ S).T + T).astype(np.float32)
        aabb_np = self.aabb.cpu().numpy()

        # Convert back to normalized space before storing in self._xyz
        return (world_transformed - aabb_np[:3]) / aabb_np[3:]
    
    def batch_compose_rs(self, R2, S2, r1, s1):
        w, x, y, z = r1.T  # (4, n)
        R1 = Rotation.from_quat(np.stack([x, y, z, w], axis=-1)).as_matrix()
        S1 = np.eye(3) * s1[..., np.newaxis]

        R2S2 = R2 @ S2
        R1S1 = np.einsum('bij,bjk->bik', R1, S1)
        RS = np.einsum('ij,bjk->bik', R2S2, R1S1)
        return RS

    def batch_decompose_rs(self, RS):
        sx = np.linalg.norm(RS[..., 0], axis=-1)
        sy = np.linalg.norm(RS[..., 1], axis=-1)
        sz = np.linalg.norm(RS[..., 2], axis=-1)

        RS[..., 0] /= sx[..., np.newaxis]
        RS[..., 1] /= sy[..., np.newaxis]
        RS[..., 2] /= sz[..., np.newaxis]
        x, y, z, w = Rotation.from_matrix(RS).as_quat().T
        r = np.stack([w, x, y, z], axis=-1)
        s = np.stack([sx, sy, sz], axis=-1)
        return r, s

    def batch_rotate_sh(self, R, shs_in, max_sh_degree=3):
        # shs_in: (n, 3, deg)
        # SH is in yzx order so here shift the order of rot mat
        rot_fn = SHRotator(R, deg=max_sh_degree)
        shs_out = np.stack([
            rot_fn(shs_in[..., 0, :]),
            rot_fn(shs_in[..., 1, :]),
            rot_fn(shs_in[..., 2, :])
        ], axis=-2)
        return shs_out

    def transform_data(self, new_position: np.ndarray, rotation_matrix: np.ndarray, scale=1.0):
        xyz = self.get_xyz.detach().cpu().numpy().astype(np.float32)
        rotation = self.get_rotation.detach().cpu().numpy().astype(np.float32)

        # object to world
        S = (np.eye(3) * scale).astype(np.float32)
        R = rotation_matrix.astype(np.float32)
        T = new_position.astype(np.float32)
        self._xyz = torch.tensor(self.transform_xyz(T, R, S, xyz)).to(self.device).to(torch.float32)

        r, s = rotation.astype(np.float32), self.scaling_activation(self.get_scaling).detach().cpu().numpy().astype(np.float32)
        RS = self.batch_compose_rs(R, S, r, s).astype(np.float32)
        r, s = self.batch_decompose_rs(RS)

        # self._rotation = torch.tensor(r).to(self.device).to(torch.float32)
        self.from_rotation(torch.tensor(r).to(self.device).to(torch.float32))
        # self._scaling = self.inverse_scaling_activation(torch.tensor(s)).to(self.device).to(torch.float32)

        if self.sh_degree > 0:
            shs_out = self.batch_rotate_sh(R, self.get_features.detach().cpu().numpy(), self.sh_degree)
            self._features_dc = torch.tensor(shs_out[..., :, :1]).to(self.device)
            self._features_rest = torch.tensor(shs_out[..., :, 1:]).to(self.device)
