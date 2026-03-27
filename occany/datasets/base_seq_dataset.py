# Based on the code from MUSt3R (https://github.com/naver/must3r)
import torch
import os.path as osp
import numpy as np
from dust3r.utils.image import imread_cv2
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, transpose_to_landscape, is_good_type, view_name
from occany.utils.image_util import get_SAM2_transforms, get_SAM3_transforms
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from occany.utils.helpers import project_lidar_world2camera, get_ray_map_lsvm
from dust3r.utils.geometry import depthmap_to_camera_coordinates
import pickle
from dust3r.datasets.base.easy_dataset import EasyDataset, CatDataset_MUSt3R, MulDataset_MUSt3R, ResizedDataset_MUSt3R
from torchvision.transforms.functional import to_tensor
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.geometry import affine_inverse_np
from occany.utils.helpers import intrinsics_c2w_to_raymap_np
from dust3r.datasets.utils.transforms import SeqColorJitter, ImgNorm
import copy




class EasyDataset_OccAny(EasyDataset):
    def __add__(self, other):
        left = self.datasets if isinstance(self, CatDataset_MUSt3R) else [self]
        right = other.datasets if isinstance(other, CatDataset_MUSt3R) else [other]
        return CatDataset_MUSt3R([*left, *right])

    def __rmul__(self, factor):
        return MulDataset_MUSt3R(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset_MUSt3R(factor, self)

    def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True):
        if not (shuffle):
            raise NotImplementedError()  # cannot deal yet

        num_of_aspect_ratios = len(self._resolutions)
        min_memory_num_views = self.min_memory_num_views
        max_memory_num_views = self.max_memory_num_views
        ray_map_prob = self.ray_map_prob
        ray_map_idx = self.ray_map_idx
        return BatchedRandomSampleOccAny(self, batch_size, 
            num_of_aspect_ratios=num_of_aspect_ratios,
            min_memory_num_views=min_memory_num_views,
            max_memory_num_views=max_memory_num_views,
            ray_map_prob=ray_map_prob,
            ray_map_idx=ray_map_idx,
            world_size=world_size, rank=rank, drop_last=drop_last)




class BaseSeqDataset (BaseStereoViewDataset):

    def __init__(self, *args, ROOT, seq_pkl_name,
                 distill_model_name="SAM2", distill_img_size=None, img_size=512, 
                 base_model="must3r",
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.base_model = base_model
        self.distill_model_name = distill_model_name
        # resolve distillation image size based on model type
        if distill_img_size is None:
            if distill_model_name == "SAM2":
                distill_img_size = img_size
            elif distill_model_name == "SAM3":
                distill_img_size = 518
        self.distill_img_size = distill_img_size
        # Create SAM2 transform with specified resolution if not provided
        if distill_model_name == "SAM2":
            self.distill_img_transform = get_SAM2_transforms(resolution=self.distill_img_size)
        elif distill_model_name == "SAM3":
            self.distill_img_transform = get_SAM3_transforms(resolution=self.distill_img_size)
        else:
            raise ValueError(f"Unsupported distill_model_name: {distill_model_name}")
        self.seq_pkl_name = seq_pkl_name
        self.img_ext = ".jpg"
        self.num_views = 3
        self._load_data()
        self.is_metric_scale = True
    
    def __len__(self):
        return len(self.seqs)

    def get_stats(self):
        return f'{len(self)} seqs from {len(self.scenes)} scenes'

    def _get_views(self, seq_idx, resolution, rng):
        scene_idx, seq, ts = self.seqs[seq_idx]
        scene_name = self.scenes[scene_idx]
        
        seq_len = len(seq)
        
        # Determine the maximum possible interval between frames to ensure all 3 frames fit
        max_delta = (seq_len + 1) // 2

        # Randomly select the time interval (delta_t) between frames (at least 1, at most max_delta-1)
        delta_t = rng.integers(1, max_delta)

        # Randomly choose a start index such that three frames with spacing delta_t fit within the sequence
        first_valid = 0
        last_valid = seq_len - delta_t * 2
        i1 = rng.integers(first_valid, last_valid)
        i2 = i1 + delta_t * 2
        
        possible_deltas = [d for d in range(1, delta_t*2)]
        delta_forecast = rng.choice(possible_deltas)
        i_forecast = i1 + delta_forecast

        # Select frame ids and corresponding timestamps for the 3 views
        img1, img2, img_forecast = seq[i1], seq[i2], seq[i_forecast]
        t1, t2, t_forecast = ts[i1], ts[i2], ts[i_forecast]

 
        
        t1, t2, t_forecast = t1 - t1, t2 - t1, t_forecast - t1
        
        preprocessed_scene_dir = osp.join(self.ROOT, scene_name)

        views = []

        frames = [img1, img2, img_forecast]
        times = [t1, t2, t_forecast]

        for view_index, t in zip(frames, times):
            frame_id = self.frames[view_index]
            image = imread_cv2(osp.join(preprocessed_scene_dir, f"{frame_id}{self.img_ext}"))
            depthmap = imread_cv2(osp.join(preprocessed_scene_dir, f"{frame_id}.exr"))
            camera_params = np.load(osp.join(preprocessed_scene_dir, f"{frame_id}.npz"))

           
            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])

            # Set skew term to 0
            intrinsics[0, 1] = 0.0
            intrinsics[1, 0] = 0.0

            image, depthmap, intrinsics = self._resize_image_and_sparse_depthmap(image, depthmap, intrinsics, resolution, rng, info=(scene_name, frame_id))

            
            views.append(dict(
                img=image,
                timestep=t,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset=self.__class__.__name__,
                scene_name=scene_name,
                frame_id=frame_id,
                label=f"{scene_name}_{frame_id}",
                instance=frame_id))

        return views

    def _resize_image_and_sparse_depthmap(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        image, _, intrinsics2 = self._crop_resize_if_necessary(
            image, depthmap, intrinsics, resolution, rng, info)

        pts3d_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, intrinsics)
        pts3d_cam = pts3d_cam[valid_mask]


        depthmap2, _, _, _ = project_lidar_world2camera(pts3d_cam,
                                                        img_h=image.height, img_w=image.width,
                                                        camera_pose=np.eye(4),
                                                        cam_K=intrinsics2)

        return image, depthmap2.astype(np.float32), intrinsics2.astype(np.float32)

    def _load_data(self):
        assert self.seq_pkl_name is not None, "seq_pkl_name must be provided"

        with open(osp.join(self.ROOT, self.seq_pkl_name), 'rb') as f:
            data = pickle.load(f)
            self.scenes = data['scenes']
            self.frames = data['frames']
            self.seqs = data['seqs']
            # Filter sequences with length less than 3
            self.seqs = [seq for seq in self.seqs if len(seq[1]) >= 3]

        print(f'Loaded {self.get_stats()}')

    def select_scene(self, scene, *instances, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        try:
            scene_id = {self.scenes.index(s) for s in scenes}
        except ValueError:
            raise AssertionError('no scene found')
        
        valid_seqs = [seq for seq in self.seqs if seq[0] in scene_id]
        
        if opposite:
            valid_seqs = [seq for seq in self.seqs if seq[0] not in scene_id]
            
        print(f"Selected {len(valid_seqs)} seqs from {len(self.seqs)}")
        self.seqs = valid_seqs

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            # view['idx'] = (idx, ar_idx, v)
            view['is_metric_scale'] = self.is_metric_scale
            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['distill_img'] = self.distill_img_transform(view['img'])
            if self.base_model == 'da3':
                
                view['img'] = InputProcessor.NORMALIZE(to_tensor(view['img']))
            else:
                view['img'] = self.transform(view['img'])


            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            view['z_far'] = self.z_far
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
            
            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            
            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']


        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views

class BaseSeqDatasetMultiView(BaseSeqDataset, EasyDataset_OccAny):
    def __init__(self, 
        ray_map_prob=0.0,
        ray_map_idx=None,
        recon_view_idx=None,
        reverse_seq=False,
        shuffle_seq_prob=0.0,
        transform=ImgNorm,
        frame_interval=1,
        max_memory_num_views=10,
        min_memory_num_views=2, 
        use_surround_temporal=False,
        *args, **kwargs):
        self.max_memory_num_views = max_memory_num_views
        self.min_memory_num_views = min_memory_num_views
        self.use_surround_temporal = use_surround_temporal
        
        super().__init__(*args, **kwargs)
        self.reverse_seq = reverse_seq
        self.shuffle_seq_prob = shuffle_seq_prob
        self.recon_view_idx = recon_view_idx
        self.ray_map_prob = ray_map_prob
        self.ray_map_idx = [] if ray_map_idx is None else list(ray_map_idx)
        self.frame_interval = frame_interval
        print(f"{self.__class__.__name__}: reverse_seq={self.reverse_seq}, shuffle_seq_prob={self.shuffle_seq_prob}")

        self.is_seq_color_jitter = False
        if isinstance(transform, str):
            transform = eval(transform)
        if transform == SeqColorJitter:
            self.is_seq_color_jitter = True
        self.transform = transform



    @staticmethod
    def generate_numbers_with_intervals(start=0, end=50, n=10, min_interval=1, max_interval=10):
        """
        Generate n random numbers with random intervals between them using NumPy.
        Guarantees exactly n numbers within the specified range.
        
        Args:
            start: Starting number of the range
            end: Ending number of the range
            n: Number of numbers to generate
            min_interval: Minimum interval between consecutive numbers
            max_interval: Maximum interval between consecutive numbers
        
        Returns:
            NumPy array of generated numbers
        """
        min_total_range = (n - 1) * min_interval
        max_total_range = end - start
        
        total_range = np.random.randint(min_total_range, max_total_range + 1)
        min_total_intervals = (n - 1) * min_interval
        
        if min_total_intervals > total_range:
            raise ValueError(f"Cannot fit {n} numbers with minimum interval {min_interval} in range {start}-{end}")
        
        # Generate base intervals (all at minimum)
        intervals = np.full(n - 1, min_interval)
        
        # Calculate extra space available
        extra_space = total_range - min_total_intervals
        
        if extra_space > 0:
            # Distribute extra space randomly
            for _ in range(extra_space):
                # Find intervals that can still be increased
                can_increase = intervals < max_interval
                available_indices = np.where(can_increase)[0]
                
                if len(available_indices) > 0:
                    # Choose random interval to increase
                    idx = np.random.choice(available_indices)
                    intervals[idx] += 1
        
        # Generate the actual numbers
        numbers = np.zeros(n, dtype=int)
        numbers[0] = start
        numbers[1:] = start + np.cumsum(intervals)
        
        return numbers


    def _get_views(self, seq_idx, resolution, memory_num_views, rng):
        scene_idx, seq, ts = self.seqs[seq_idx]
        scene_name = self.scenes[scene_idx]
        preprocessed_scene_dir = osp.join(self.ROOT, scene_name)
        
        seq_len = len(seq)
  
        # If memory_num_views is larger than seq_len, sample with replacement (repeat frames)
        if seq_len < memory_num_views:
            # Sample all available frames and then repeat some randomly
            memory_view_indices = list(range(seq_len))
            # Need to add (memory_num_views - seq_len) more frames by repeating
            num_repeats = memory_num_views - seq_len
            repeated_indices = rng.choice(seq_len, size=num_repeats, replace=True)
            memory_view_indices.extend(repeated_indices.tolist())
            # Shuffle to mix repeated frames with original frames
            rng.shuffle(memory_view_indices)
        else:
            if self.use_surround_temporal:
                # For surround_temporal: sample one starting frame, then randomly sample the rest
                start_idx = 0
                remaining_indices = [i for i in range(seq_len) if i != start_idx]
                
                # Randomly sample (memory_num_views - 1) indices from the remaining frames
                if len(remaining_indices) >= memory_num_views - 1:
                    sampled_indices = rng.choice(remaining_indices, size=memory_num_views-1, replace=False)
                else:
                    # If not enough remaining frames, sample with replacement
                    sampled_indices = rng.choice(remaining_indices, size=memory_num_views-1, replace=True)
                
                memory_view_indices = [start_idx] + list(sampled_indices)
            else:
                # Normal case: sample consecutive memory_num_views frames from the sequence
                # Choose a random starting index that allows for memory_num_views consecutive frames
                start_idx = rng.integers(0, seq_len - (memory_num_views-1) * self.frame_interval)
                end_idx = start_idx + (memory_num_views-1) * self.frame_interval
                
                inbetween_idx = rng.choice(np.arange(start_idx+1, end_idx+1), size=memory_num_views-1, replace=False)

                memory_view_indices = [start_idx] + list(inbetween_idx)
        memory_view_indices = sorted(memory_view_indices)
      

        if self.reverse_seq and rng.random() < 0.5:
            memory_view_indices = memory_view_indices[::-1]
        
        # Randomly shuffle the sequence based on shuffle_seq_prob
        if self.shuffle_seq_prob > 0 and rng.random() < self.shuffle_seq_prob:
            memory_view_indices = list(rng.permutation(memory_view_indices))

    
        
        frames = [seq[i] for i in memory_view_indices]
        times = [ts[i] for i in memory_view_indices]

        views = []
        for i, (view_index, t) in enumerate(zip(frames, times)):
            frame_id = self.frames[view_index]
          
            npz_path = osp.join(preprocessed_scene_dir, f"{frame_id}.npz")
            try:
                data = np.load(npz_path)
            except Exception:
                raise RuntimeError(f"Failed to load dataset sample: {npz_path}")
    
        
            image = data['image']          # The image array
            depthmap = data['depthmap']    # The depth map
            intrinsics = np.float32(data['intrinsics']) # Camera intrinsics matrix
            camera_pose = np.float32(data['cam2world'])  # Camera-to-world transformation matrix


            # Set skew term to 0
            intrinsics[0, 1] = 0.0
            intrinsics[1, 0] = 0.0

            image, depthmap, intrinsics = self._resize_image_and_sparse_depthmap(image, depthmap, intrinsics, resolution, rng, info=(scene_name, frame_id))

           
            # else:

            views.append(dict(
                img=image,
                timestep=t,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset=self.__class__.__name__,
                scene_name=scene_name,
                frame_id=frame_id,
                label=f"{scene_name}_{frame_id}",
                instance=frame_id))
        
        return views

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, resolution_idx, memory_num_views, ray_map_idx = idx
        else:
            # This is used by test data as we don't implement the BatchSampler in test 
            assert len(self._resolutions) == 1
            resolution_idx = 0
            assert self.min_memory_num_views == self.max_memory_num_views, "Evaluation needs to be done with a fixed number of views, which is equal to min_memory_num_views and  min_memory_num_views must equal max_memory_num_views"
            memory_num_views = self.min_memory_num_views
            # assert len(self.ray_map_idx) != 0, "Evaluation needs to be done with fixed ray_map_idx"
            ray_map_idx = self.ray_map_idx
        
        assert all(ray_map_id < memory_num_views for ray_map_id in ray_map_idx), f"ray_map_idx should be smaller than memory_num_views ray_map_idx={ray_map_idx}, memory_num_views={memory_num_views}"
        # idx, ar_idx, memory_num_views = 290, 0, 10 # TODO: remove later as this is only for overfitting 1 example
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded codez_far
        resolution = self._resolutions[resolution_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, memory_num_views, self._rng)
        
        
        transform = SeqColorJitter() if self.is_seq_color_jitter else self.transform

        in_camera0 = affine_inverse_np(views[0]['camera_pose'])
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, resolution_idx, v, memory_num_views, ray_map_idx)
            view['memory_num_views'] = memory_num_views
            view['is_metric_scale'] = self.is_metric_scale
            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['distill_img'] = self.distill_img_transform(view['img'])
            if self.base_model == 'da3':
                view['img'] = InputProcessor.NORMALIZE(to_tensor(view['img']))
            else:
                view['img'] = transform(view['img'])
            
            # convert all camera poses to the coordinate frame of camera 0
            view['camera_pose'] = in_camera0 @ view['camera_pose']

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            view['z_far'] = self.z_far

            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            
            # Generate GT raymap that matches pts3d computation
            gt_raymap = intrinsics_c2w_to_raymap_np(
                view['camera_intrinsics'],  # (3, 3)
                view['camera_pose'],        # (4, 4) - camera-to-world
                height,
                width,
            )  # (H, W, 6)
            view['gt_raymap'] = gt_raymap
            pts3d_from_raymap = view['depthmap'][..., None] * gt_raymap[..., :3] + gt_raymap[..., 3:]
            if valid_mask.any():
                max_diff = np.abs((pts3d - pts3d_from_raymap)[valid_mask]).max()
                assert max_diff < 1e-3, f"pts3d mismatch: max_diff={max_diff:.6e} for view {view_name(view)}"
            
            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

        if self.recon_view_idx is None:
            recon_view_idx = [i for i in range(len(views)) if i not in ray_map_idx]
            # use the dataset's RNG for reproducibility across workers
            gen_view_idx = ray_map_idx
        else:
            # only during test that we explicit pass recon_view_idx
            gen_view_idx = ray_map_idx
            recon_view_idx = self.recon_view_idx

        assert recon_view_idx[0] == 0, f"recon_view_idx={recon_view_idx}"
        camera_poses = np.stack([view['camera_pose'] for view in views], axis=0)
        ret_views = []
        for v in recon_view_idx:
            view = copy.deepcopy(views[v])
            view['is_raymap'] = False
            ret_views.append(view)
        
        for v in gen_view_idx:
            view = copy.deepcopy(views[v])
            # get ray map
            camera_pose = torch.from_numpy(camera_poses[v])[None, None, :, :]
            fxfycxcy = torch.zeros((1, 1, 4), dtype=torch.float32)
            K_torch = torch.from_numpy(K)
            fxfycxcy[:, :, 0] = K_torch[0, 0]
            fxfycxcy[:, :, 1] = K_torch[1, 1]
            fxfycxcy[:, :, 2] = K_torch[0, 2]
            fxfycxcy[:, :, 3] = K_torch[1, 2]
            
            ray_map_lsvm = get_ray_map_lsvm(camera_pose, fxfycxcy, h=height, w=width)
            ray_map_lsvm = ray_map_lsvm.squeeze()
          
            ray_map_mask = np.array([1.0], dtype=np.float32)
            view['ray_map'] = ray_map_lsvm
            # view['ray_map_mask'] = ray_map_mask
            view['is_raymap'] = True
            ret_views.append(view)
        
        # last thing done!
        for view in ret_views:
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
     
        
        return ret_views

