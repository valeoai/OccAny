import argparse
import glob
import os
from tqdm import tqdm
import pickle


class SeqMaker:
    def __init__(self,
                 preprocessed_dir, cameras,
                 img_track_pattern="*_{camera_id}",
                 frame_id_format=":06d",
                 prefix="",
                 suffix="",
                 file_ext=".npz", subsampling_rate=1, max_stride=9,
                 seq_mode="temporal"):
        """
        Assume 10Hz data
        Args:
            preprocessed_dir: Directory containing preprocessed data
            cameras: List of camera IDs
            img_track_pattern: Pattern for image filenames with {camera_id} placeholder
            frame_id_format: Format string for frame ID (e.g. ":06d" for 6-digit zero-padded numbers)
            file_ext: File extension for RGB images (e.g. ".jpg")
            subsampling_rate: Rate at which to subsample frames
            max_stride: Maximum stride between frames
            seq_mode: "temporal" for sequences over time (same camera), "surround" for sequences across cameras (same time)
        """
        self.subsampling_rate = subsampling_rate
        self.max_stride = max_stride
        self.processed_root = self._get_processed_root(preprocessed_dir)
        self.scenes = []
        self.frames = []
        self.seqs = []
        self.cameras = cameras
        self.img_track_pattern = img_track_pattern
        self.frame_id_format = frame_id_format
        self.file_ext = file_ext
        self.prefix = prefix
        self.suffix = suffix
        self.seq_mode = seq_mode


    def _get_processed_root(self, preprocessed_dir):
        assert "SCRATCH" in os.environ, "SCRATCH environment variable is not set"
        SCRATCH = os.environ["SCRATCH"]
        return os.path.join(SCRATCH, f"data/{preprocessed_dir}")

    def _load_scenes_and_frames(self):
        print("Loading scenes and frames...")
        self.scenes = [d for d in os.listdir(self.processed_root) if os.path.isdir(os.path.join(self.processed_root, d)) and "tmp" not in d]
        self.scenes.sort()
        print(f"Loaded {len(self.scenes)} scenes")
        frames_set = set()
        pbar = tqdm(self.scenes, desc="Loading frames")
        for scene in pbar:
            path = os.path.join(self.processed_root, scene)
            imgs = set()
            for camera_id in self.cameras:
                pattern = self.img_track_pattern.format(camera_id=camera_id) + self.file_ext
                imgs.update(glob.glob(os.path.join(path, pattern)))
            frames_set.update(img.split('/')[-1].replace(self.file_ext, '') for img in imgs)
            pbar.set_postfix_str(f"Scene {scene}: unique frames {len(frames_set)}")

        self.frames = sorted(list(frames_set))
        print(f"Loaded {len(self.frames)} frames")

    def _generate_seq(self):
        # strides = [s * self.subsampling_rate for s in range(0, self.max_stride * 2 + 1)]
        strides = [s * self.subsampling_rate for s in range(0, self.max_stride+1)]
        print(f"Strides: {strides}")
        frame_to_index = {frame: idx for idx, frame in enumerate(self.frames)}
        scene_to_index = {scene: idx for idx, scene in enumerate(self.scenes)}

        pbar = tqdm(self.scenes, desc="Generating seq")
        for scene_name in pbar:
            scene_idx = scene_to_index[scene_name]
            path = os.path.join(self.processed_root, scene_name)

            file_tracks = []
            for camera_id in self.cameras:
                pattern = self.img_track_pattern.format(camera_id=camera_id) + self.file_ext
                file_track = sorted(glob.glob(os.path.join(path, pattern)))
                file_tracks.append(file_track)

            # for stride in strides:
            for file_track in file_tracks:
                for i in range(len(file_track)):
                    current_file_path = file_track[i]
                    current_file_name = current_file_path.split('/')[-1].replace(self.file_ext, '')

                    frame_id_str, cam_id = current_file_name.split('_')
                    frame_id = int(frame_id_str)


                    seq = []
                    timesteps = []
                    for stride in strides:
                        next_frame_id = f"{(frame_id + stride):{self.frame_id_format[1:]}}"
                        next_frame_name = f"{next_frame_id}_{cam_id}"
                        next_frame_path = os.path.join(path, f"{next_frame_name}{self.file_ext}")
                        if os.path.exists(next_frame_path):
                            next_frame_idx = frame_to_index[next_frame_name]
                            seq.append(next_frame_idx)
                            timesteps.append(stride)
                    if len(seq) == len(strides):
                        self.seqs.append([scene_idx, seq, timesteps])
            pbar.set_postfix({"seqs": len(self.seqs)})

    def _generate_surround_seq(self):
        """Generate sequences across different cameras at the same timestep (surround view)"""
        print(f"Generating surround view sequences for {len(self.cameras)} cameras")
        frame_to_index = {frame: idx for idx, frame in enumerate(self.frames)}
        scene_to_index = {scene: idx for idx, scene in enumerate(self.scenes)}

        pbar = tqdm(self.scenes, desc="Generating surround seq")
        for scene_name in pbar:
            scene_idx = scene_to_index[scene_name]
            path = os.path.join(self.processed_root, scene_name)

            # Get all frames for each camera
            file_tracks = []
            for camera_id in self.cameras:
                pattern = self.img_track_pattern.format(camera_id=camera_id) + self.file_ext
                file_track = sorted(glob.glob(os.path.join(path, pattern)))
                file_tracks.append(file_track)

            # For each frame in the first camera, try to find matching frames in all other cameras
            if not file_tracks:
                continue
            
            for file_path in file_tracks[0]:
                file_name = file_path.split('/')[-1].replace(self.file_ext, '')
                frame_id_str, cam_id = file_name.split('_')
                frame_id = int(frame_id_str)

                seq = []
                camera_ids = []
                
                # Try to find this frame ID in all cameras
                for camera_id in self.cameras:
                    frame_name = f"{frame_id_str}_{camera_id}"
                    frame_path = os.path.join(path, f"{frame_name}{self.file_ext}")
                    
                    if os.path.exists(frame_path) and frame_name in frame_to_index:
                        frame_idx = frame_to_index[frame_name]
                        seq.append(frame_idx)
                        camera_ids.append(camera_id)
                # Only add sequence if all cameras have this frame
                if len(seq) == len(self.cameras):
                    # For surround view, timesteps are all 0 (same time)
                    timesteps = [0] * len(seq)
                    self.seqs.append([scene_idx, seq, timesteps])
            
            pbar.set_postfix({"seqs": len(self.seqs)})

    def save_seq(self):
        print(len(self.seqs), "seq generated.")
        if self.seq_mode == "surround":
            save_filename = f"{self.prefix}seq_surround{self.suffix}.pkl"
        else:
            save_filename = f"{self.prefix}seq_exact_len_sub{self.subsampling_rate}_stride{self.max_stride}{self.suffix}.pkl"
        save_path = os.path.join(self.processed_root, save_filename)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'scenes': self.scenes,
                'frames': self.frames,
                'seqs': self.seqs
            }, f)
        print(f"Saved to {save_path}")

    def run(self):

        self._load_scenes_and_frames()
        print(f"Generating seq in {self.seq_mode} mode...")
        if self.seq_mode == "surround":
            self._generate_surround_seq()
        else:
            self._generate_seq()
        self.save_seq()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create image pairs.")
    parser.add_argument('--prefix', type=str, default="", help='Optional prefix for the output PKL filename')
    parser.add_argument('--subsampling_rate', type=int, default=5, help='Subsampling rate for image pairs')
    parser.add_argument('--max_stride', type=int, default=2, help='Maximum stride for image pairs at subsampling_rate')
    parser.add_argument('--dataset', type=str, default="ddad", help='Dataset name')
    parser.add_argument('--camera', type=str, choices=["all", "surround"], default="all",
                        help='Camera ID (all for temporal mode, or surround)')
    parser.add_argument('--seq_mode', type=str, choices=["temporal", "surround"], default="temporal", 
                        help='Sequence mode: temporal (same camera, different times) or surround (different cameras, same time)')
    return parser.parse_args()


def resolve_cameras(dataset, camera, seq_mode):
    if camera == "all" and seq_mode != "temporal":
        raise ValueError("Camera 'all' is only supported with temporal sequence generation")

    if dataset == "waymo":
        camera_map = {
            "surround": list(range(1, 6)),
            "all": list(range(1, 6)),
        }
    elif dataset == "ddad":
        camera_map = {
            "surround": list(range(6)),
            "all": list(range(6)),
        }
    elif dataset == "pandaset":
        camera_map = {
            "surround": list(range(6)),
            "all": list(range(6)),
        }
    elif dataset == "once":
        camera_map = {
            "surround": ["cam01", "cam05", "cam06", "cam07", "cam08"],
            "all": ["cam06", "cam07", "cam08", "cam09"],
        }
    else:
        raise ValueError(f"Dataset {dataset} does not define camera presets")

    try:
        return camera_map[camera]
    except KeyError as exc:
        raise ValueError(f"Camera {camera} not supported for {dataset}") from exc


def resolve_sequence_suffix(camera, seq_mode):
    if camera == "surround" and seq_mode == "surround":
        return "_all"
    return f"_{camera}"


if __name__ == "__main__":
    args = parse_arguments()

    if args.dataset == "waymo":
        cameras = resolve_cameras(args.dataset, args.camera, args.seq_mode)
        seq_maker = SeqMaker(preprocessed_dir="waymo_processed",
                                     cameras=cameras,
                                     img_track_pattern="*_{camera_id}",
                                     frame_id_format=":05d",
                                     prefix=args.prefix,
                                     suffix=resolve_sequence_suffix(args.camera, args.seq_mode),
                                     subsampling_rate=args.subsampling_rate,
                                     max_stride=args.max_stride,
                                     seq_mode=args.seq_mode)
    elif args.dataset == "vkitti":
        seq_maker = SeqMaker(
            preprocessed_dir="vkitti_processed",
            cameras=[0],
            img_track_pattern="*_{camera_id}",
            frame_id_format=":05d",
            prefix=args.prefix,
            subsampling_rate=args.subsampling_rate,
            max_stride=args.max_stride,
            seq_mode=args.seq_mode
        )
    elif args.dataset == "ddad":
        cameras = resolve_cameras(args.dataset, args.camera, args.seq_mode)
        seq_maker = SeqMaker(preprocessed_dir="ddad_processed",
                                     cameras=cameras,
                                     img_track_pattern="*_{camera_id}",
                                     frame_id_format=":06d",
                                     prefix=args.prefix,
                                     suffix=resolve_sequence_suffix(args.camera, args.seq_mode),
                                     subsampling_rate=args.subsampling_rate,
                                     max_stride=args.max_stride,
                                     seq_mode=args.seq_mode)
    elif args.dataset == "pandaset":
        cameras = resolve_cameras(args.dataset, args.camera, args.seq_mode)
        seq_maker = SeqMaker(preprocessed_dir="pandaset_processed",
                                     cameras=cameras,
                                     img_track_pattern="*_{camera_id}",
                                     frame_id_format=":06d",
                                     prefix=args.prefix,
                                     suffix=resolve_sequence_suffix(args.camera, args.seq_mode),
                                     subsampling_rate=args.subsampling_rate,
                                     max_stride=args.max_stride,
                                     seq_mode=args.seq_mode)
    elif args.dataset == "kitti":
        seq_maker = SeqMaker(preprocessed_dir="kitti_processed",
                                     cameras=[0],
                                     img_track_pattern="*_{camera_id}",
                                     frame_id_format=":06d",
                                     file_ext=".npz",
                                           prefix=args.prefix,
                                           subsampling_rate=args.subsampling_rate,
                                           max_stride=args.max_stride,
                                           seq_mode=args.seq_mode)
    elif args.dataset == "once":
        cameras = resolve_cameras(args.dataset, args.camera, args.seq_mode)
        seq_maker = SeqMaker(preprocessed_dir="once_processed",
                                     cameras=cameras,
                                     img_track_pattern="*_{camera_id}",
                                     frame_id_format=":06d",
                                     prefix=args.prefix,
                                     suffix=resolve_sequence_suffix(args.camera, args.seq_mode),
                                     subsampling_rate=args.subsampling_rate,
                                     max_stride=args.max_stride,
                                     seq_mode=args.seq_mode)
    elif args.dataset == "occ3d_nuscenes_all":
        if args.camera == "surround":
            cameras = [0, 1, 2, 3, 4, 5]  # All 6 nuScenes cameras
        else:
            raise ValueError(f"Camera {args.camera} not supported for occ3d_nuscenes (only 'surround' is supported)")
     
        seq_maker = SeqMaker(preprocessed_dir="occ3d_nuscenes_processed_all",
                                     cameras=cameras,
                                     img_track_pattern="*_{camera_id}",
                                     frame_id_format=":06d",
                                     file_ext=".npz",
                                     prefix=args.prefix,
                                     suffix=resolve_sequence_suffix(args.camera, args.seq_mode),
                                     subsampling_rate=args.subsampling_rate,
                                     max_stride=args.max_stride,
                                     seq_mode=args.seq_mode)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    seq_maker.run()
