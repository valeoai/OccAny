import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
import viser
import time
from scipy.spatial.transform import Rotation as R
import cv2

from occany.utils.vis_util import OCC3D_COLORS


OCC3D_RGB_COLORS = OCC3D_COLORS[:, :3].astype(np.uint8, copy=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./demo_data/output",
        help=(
            "Path to a demo output root that contains scene folders, "
            "or a single scene folder containing pts3d_*.npy files."
        ),
    )
    return parser.parse_args()


def get_kitti_color_map():
    """Return KITTI class ID to RGB color mapping."""
    return np.array([
        [0, 0, 0, 255], # "empty"
        [100, 150, 245, 255], # "car"
        [100, 230, 245, 255], # "bicycle"
        [30, 60, 150, 255], # "motorcycle"
        [80, 30, 180, 255], # "truck"
        [100, 80, 250, 255], # "other-vehicle"
        [255, 30, 30, 255], # "person"
        [255, 40, 200, 255], # "rider"
        [150, 30, 90, 255], # "motorcyclist"
        [255, 0, 255, 255], # "road"
        [255, 150, 255, 255], # "parking"
        [75, 0, 75, 255], # "sidewalk"
        [175, 0, 75, 255], # "other-ground"
        [255, 200, 0, 255], # "building"
        [255, 120, 50, 255], # "fence"
        [0, 175, 0, 255], # "vegetation"
        [135, 60, 0, 255], # "trunk"
        [150, 240, 80, 255], # "terrain"
        [255, 240, 150, 255], # "pole"
        [255, 0, 0, 255], # "traffic-sign"
        [169, 169, 169, 255], # "unknown"
    ])[:, :3].astype(np.uint8)


def get_nuscenes_color_map():
    """Return NuScenes class ID to RGB color mapping."""
    return np.array([
        [0, 0, 0, 255],
        [112, 128, 144, 255],
        [220, 20, 60, 255],
        [255, 127, 80, 255],
        [255, 158, 0, 255],
        [233, 150, 70, 255],
        [255, 61, 99, 255],
        [0, 0, 230, 255],
        [47, 79, 79, 255],
        [255, 140, 0, 255],
        [255, 98, 70, 255],
        [0, 207, 191, 255],
        [175, 0, 75, 255],
        [75, 0, 75, 255],
        [112, 180, 60, 255],
        [222, 184, 135, 255],
        [0, 175, 0, 255],
        [135, 206, 235, 255], # sky, empty
    ])[:, :3].astype(np.uint8)


def get_available_settings(scene_dir: str) -> tuple[str, ...]:
    settings = []
    for entry in sorted(os.listdir(scene_dir)):
        file_path = os.path.join(scene_dir, entry)
        if os.path.isfile(file_path) and entry.startswith("pts3d_") and entry.endswith(".npy"):
            settings.append(entry[len("pts3d_"):-4])
    return tuple(settings)


def discover_scene_dirs(input_folder: str) -> dict[str, str]:
    input_folder = os.path.abspath(input_folder)
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    direct_settings = get_available_settings(input_folder)
    if direct_settings:
        scene_name = os.path.basename(os.path.normpath(input_folder))
        return {scene_name: input_folder}

    scene_dirs = {}
    for entry in sorted(os.listdir(input_folder)):
        scene_dir = os.path.join(input_folder, entry)
        if not os.path.isdir(scene_dir) or entry == "saved_views":
            continue
        if get_available_settings(scene_dir):
            scene_dirs[entry] = scene_dir

    if not scene_dirs:
        raise RuntimeError(
            f"No scene folders with pts3d_*.npy files were found in {input_folder}"
        )

    return scene_dirs


def choose_default_setting(available_settings: tuple[str, ...]) -> str:
    if not available_settings:
        raise RuntimeError("No pts3d_*.npy files were found for the selected scene.")
    return "render" if "render" in available_settings else available_settings[0]


def load_data(scene_dir, setting):
    file_dir = os.path.join(scene_dir, f"pts3d_{setting}.npy")
    save_dict = np.load(file_dir, allow_pickle=True).item()
    
    
    
    for i, color in enumerate(save_dict['colors']):
        if color.sum() == 0:
            y_inverse = -save_dict['pts3d'][i, :, :, 1]
            # Compute global min/max for consistent coloring across all views
            dim = 1  # Y-axis (height)
            y_min = max(-2.0, y_inverse.min())
            y_max = min(1.5, y_inverse.max())
            
            # Avoid division by zero
            if y_max - y_min < 1e-6:
                y_max = y_min + 1.0
            normalized = ((y_inverse - y_min) / (y_max - y_min)).clip(0.0, 1.0)
            # Convert to uint8 for OpenCV colormap (0-255 range)
            normalized_uint8 = (normalized * 255).astype(np.uint8)
            # Apply OpenCV colormap (returns BGR format)
            colors_bgr = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
            # Convert BGR to RGB
            colors_rgb = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)
            # Convert to float (-1.0 to 1.0 range as expected by the rest of the code)
            colors_np = (colors_rgb.astype(np.float32) / 127.5) - 1.0
            save_dict['colors'][i] = colors_np

    return save_dict


def compute_height_colors(pts3d: np.ndarray, y_min_clip=-2.0, y_max_clip=1.5, colormap='JET', invert_y=True) -> np.ndarray:
    """
    Compute height-based colors for all points using a colormap.
    
    Args:
        pts3d: Point cloud array of shape (N, H, W, 3)
        y_min_clip: Minimum Y value to clip (for normalization)
        y_max_clip: Maximum Y value to clip (for normalization)
        colormap: OpenCV colormap name (e.g., 'JET', 'PLASMA', 'VIRIDIS', 'TURBO')
        invert_y: If True, use -Y for coloring (useful when Y-axis points down)
    
    Returns:
        colors: Array of shape (N, H, W, 3) with colors in [-1, 1] range
    """
    # Map colormap name to OpenCV constant
    colormap_dict = {
        'JET': cv2.COLORMAP_JET,
        'PLASMA': cv2.COLORMAP_PLASMA,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'TURBO': cv2.COLORMAP_TURBO,
        'HOT': cv2.COLORMAP_HOT,
        'COOL': cv2.COLORMAP_COOL,
        'RAINBOW': cv2.COLORMAP_RAINBOW,
    }
    colormap_cv = colormap_dict.get(colormap.upper(), cv2.COLORMAP_JET)
    
    N, H, W, _ = pts3d.shape
    colors = np.zeros((N, H, W, 3), dtype=np.float32)
    
    for i in range(N):
        # Get Y-axis values (height)
        y_values = pts3d[i, :, :, 1]
        if invert_y:
            y_values = -y_values
        
        # Compute min/max with clipping
        y_min = max(y_min_clip, y_values.min())
        y_max = min(y_max_clip, y_values.max())
        
        # Avoid division by zero
        if y_max - y_min < 1e-6:
            y_max = y_min + 1.0
        
        # Normalize to [0, 1]
        normalized = ((y_values - y_min) / (y_max - y_min)).clip(0.0, 1.0)
        
        # Convert to uint8 for OpenCV colormap (0-255 range)
        normalized_uint8 = (normalized * 255).astype(np.uint8)
        
        # Apply OpenCV colormap (returns BGR format)
        colors_bgr = cv2.applyColorMap(normalized_uint8, colormap_cv)
        
        # Convert BGR to RGB
        colors_rgb = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to float (-1.0 to 1.0 range)
        colors[i] = (colors_rgb.astype(np.float32) / 127.5) - 1.0
    
    return colors


def compute_semantic_colors(semantic_2ds: np.ndarray, exp_dir: str = None) -> np.ndarray:
    """Convert semantic class IDs into RGB colors in the [-1, 1] range using the shared OCC3D palette."""

    if semantic_2ds is None:
        raise ValueError("Semantic array is required to compute semantic colors.")

    semantic_array = np.asarray(semantic_2ds)

    if semantic_array.ndim == 4 and semantic_array.shape[-1] == 1:
        semantic_array = semantic_array[..., 0]
    if semantic_array.ndim == 2:
        semantic_array = semantic_array[None, ...]
    if semantic_array.ndim != 3:
        raise ValueError(f"Unsupported semantic shape {semantic_array.shape}; expected (N, H, W) or (H, W).")

    semantic_array = np.nan_to_num(
        semantic_array,
        nan=len(OCC3D_RGB_COLORS) - 1,
        posinf=len(OCC3D_RGB_COLORS) - 1,
        neginf=0.0,
    ).astype(np.int64, copy=False)

    n_frames, H, W = semantic_array.shape

    colors = np.zeros((n_frames, H, W, 3), dtype=np.float32)
    for i in range(n_frames):
        semantic_frame = np.clip(semantic_array[i], 0, len(OCC3D_RGB_COLORS) - 1)
        colors[i] = OCC3D_RGB_COLORS[semantic_frame]

    return (colors.astype(np.float32) / 127.5) - 1.0


def compute_confidence_colors(conf: np.ndarray) -> np.ndarray:
    """Convert confidence scores into RGB colors in the [-1, 1] range."""

    if conf is None:
        raise ValueError("Confidence array is required to compute confidence colors.")

    conf_array = np.asarray(conf, dtype=np.float32)
    conf_array = np.nan_to_num(conf_array, nan=0.0, posinf=0.0, neginf=0.0)

    if conf_array.ndim == 4 and conf_array.shape[-1] == 1:
        conf_array = conf_array[..., 0]
    if conf_array.ndim == 2:
        conf_array = conf_array[None, ...]
    if conf_array.ndim != 3:
        raise ValueError(f"Unsupported confidence shape {conf_array.shape}; expected (N, H, W) or (H, W).")

    conf_min = float(conf_array.min()) if conf_array.size else 0.0
    conf_max = float(conf_array.max()) if conf_array.size else 1.0
    if (not np.isfinite(conf_min)) or (not np.isfinite(conf_max)) or conf_max <= conf_min:
        conf_min, conf_max = 0.0, 1.0

    conf_norm = np.clip((conf_array - conf_min) / max(conf_max - conf_min, 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap('RdYlGn')
    colors = cmap(conf_norm)[..., :3].astype(np.float32)
    return (colors * 2.0) - 1.0


def compute_conf_slider_params(conf: np.ndarray) -> tuple[float, float, float]:
    conf_array = np.asarray(conf, dtype=np.float32).reshape(-1)
    conf_array = conf_array[np.isfinite(conf_array)]
    if conf_array.size == 0:
        return 0.0, 1.0, 0.01

    conf_min = float(conf_array.min())
    conf_max = float(conf_array.max())
    if conf_max < conf_min:
        conf_min, conf_max = conf_max, conf_min

    conf_range = conf_max - conf_min
    conf_step = 1.0 if conf_range <= 0 else max(conf_range / 100.0, 1e-4)
    return conf_min, conf_max, conf_step


def draw_scene(server: viser.ViserServer, pts3d: np.ndarray,
    colors: np.ndarray, conf: np.ndarray, conf_threshold: float,
    c2w: np.ndarray, focal: np.ndarray, H: int, W: int,
    scene_handles: dict[str, list], gui_point_size) -> None:
    """Draw the point clouds and camera frustums for the current frame set once."""

    # Remove previously drawn geometry before adding the new scene.
    for handle in scene_handles.get("point_clouds", []):
        try:
            handle.remove()
        except Exception:
            pass
    scene_handles.setdefault("point_clouds", []).clear()

    for handle in scene_handles.get("frustums", []):
        try:
            handle.remove()
        except Exception:
            pass
    scene_handles.setdefault("frustums", []).clear()

    for i in range(pts3d.shape[0]):
        pts3d_frame = pts3d[i].reshape(-1, 3)
        colors_frame_uint8 = ((colors[i] + 1.0) / 2.0 * 255).astype(np.uint8)
        colors_frame_flat = colors_frame_uint8.reshape(-1, 3)

        conf_frame = np.asarray(conf[i])
        if conf_frame.ndim == 3 and conf_frame.shape[-1] == 1:
            conf_frame = conf_frame[..., 0]
        conf_mask = conf_frame.reshape(-1) >= conf_threshold

        if np.any(conf_mask):
            pts3d_i = pts3d_frame[conf_mask]
            colors_i = colors_frame_flat[conf_mask]

            point_handle = server.scene.add_point_cloud(
                name=f"/points/pts3d_{i}",
                points=pts3d_i,
                colors=colors_i,
                point_size=gui_point_size.value,
                point_shape="rounded",
            )
            scene_handles["point_clouds"].append(point_handle)

        rotation_matrix = c2w[i][:3, :3]
        position = c2w[i][:3, 3]
        # Convert rotation matrix to quaternion in wxyz format for viser
        q_xyzw = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
        rotation_quaternion = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # [w, x, y, z]
        fov = 2 * np.arctan2(H / 2, focal[i])
        aspect_ratio = W / H
        frustum_scale = 1
  
    
        frustum_handle = server.scene.add_camera_frustum(
            name=f"/cams/t{i}",
            fov=np.deg2rad(60.0),
            aspect=aspect_ratio,
            scale=frustum_scale,
            color=(255, 0, 0),
            image=colors_frame_uint8,
            wxyz=rotation_quaternion,
            position=position,
            visible=True,
        )
        scene_handles["frustums"].append(frustum_handle)


def main():
    args = get_args()
    input_folder = os.path.abspath(args.input_folder)
    scene_dirs = discover_scene_dirs(input_folder)
    scenes = tuple(scene_dirs.keys())
    current_scene = scenes[0]
    current_scene_dir = scene_dirs[current_scene]

    available_settings = get_available_settings(current_scene_dir)
    setting = choose_default_setting(available_settings)
    save_dict = load_data(current_scene_dir, setting)

    print(f"Input folder: {input_folder}")
    print(f"Loaded scenes: {', '.join(scenes)}")

    pts3d = save_dict['pts3d']
    colors = save_dict['colors']
    conf = save_dict['conf']
    focal = save_dict['focal']
    c2w = save_dict['c2w']

    conf_colors = compute_confidence_colors(conf)
    
    # Compute height-based colors
    height_colors = compute_height_colors(pts3d, y_min_clip=-2.0, y_max_clip=1.5, colormap='JET', invert_y=True)
    
    # Load semantic data if available
    if 'semantic_2ds' in save_dict:
        semantic_colors = compute_semantic_colors(save_dict['semantic_2ds'], current_scene_dir)
    else:
        semantic_colors = None
    
    conf_min, conf_max, conf_step = compute_conf_slider_params(conf)

    H, W = pts3d.shape[1], pts3d.shape[2]

    server = viser.ViserServer()

    server.scene.enable_default_lights()

    server.gui.configure_theme(
            dark_mode=False,
            show_share_button=False,
        )

        
    scene_handles = {
        "point_clouds": [],
        "frustums": [],
    }
    gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.02,
            max=0.2,
            step=0.02,
            initial_value=0.05  ,
        )

    gui_scene_dropdown = server.gui.add_dropdown(
        "Scene",
        options=scenes,
        initial_value=current_scene,
    )

    gui_setting_dropdown = server.gui.add_dropdown(
        "Setting",
        options=available_settings,
        initial_value=setting,
    )

    gui_color_mode = server.gui.add_dropdown(
        "Color source",
        ("Saved colors", "Confidence", "Semantic", "Height"),
        initial_value="Saved colors",
    )

    gui_conf_threshold = server.gui.add_slider(
        "Confidence threshold",
        min=conf_min,
        max=conf_max,
        step=conf_step,
        initial_value=conf_min,
    )

    gui_height_min = server.gui.add_slider(
        "Height min (Y)",
        min=-5.0,
        max=2.0,
        step=0.1,
        initial_value=-2.0,
    )

    gui_height_max = server.gui.add_slider(
        "Height max (Y)",
        min=-2.0,
        max=5.0,
        step=0.1,
        initial_value=1.5,
    )
    
    # Add save image button
    gui_save_button = server.gui.add_button("Save Current View")

    def refresh_conf_slider_bounds():
        nonlocal conf_min, conf_max, conf_step
        conf_min, conf_max, conf_step = compute_conf_slider_params(conf)
        gui_conf_threshold.min = conf_min
        gui_conf_threshold.max = conf_max
        gui_conf_threshold.step = conf_step
        gui_conf_threshold.value = min(max(gui_conf_threshold.value, conf_min), conf_max)

    # Rendering control: draw once on startup.
    need_redraw = True

    @gui_point_size.on_update
    def _(_):
        nonlocal need_redraw
        need_redraw = True

    @gui_color_mode.on_update
    def _(_):
        nonlocal need_redraw
        need_redraw = True

    @gui_conf_threshold.on_update
    def _(_):
        nonlocal need_redraw
        need_redraw = True

    @gui_height_min.on_update
    def _(_):
        nonlocal need_redraw, height_colors
        # Recompute height colors with new min value
        height_colors = compute_height_colors(
            pts3d, 
            y_min_clip=gui_height_min.value, 
            y_max_clip=gui_height_max.value, 
            colormap='JET', 
            invert_y=True
        )
        need_redraw = True

    @gui_height_max.on_update
    def _(_):
        nonlocal need_redraw, height_colors
        # Recompute height colors with new max value
        height_colors = compute_height_colors(
            pts3d, 
            y_min_clip=gui_height_min.value, 
            y_max_clip=gui_height_max.value, 
            colormap='JET', 
            invert_y=True
        )
        need_redraw = True
    
    @gui_save_button.on_click
    def _(event: viser.GuiEvent):
        """Save a screenshot of the current 3D view from the client's camera."""
        save_output_dir = os.path.join(input_folder, "saved_views")
        os.makedirs(save_output_dir, exist_ok=True)
        
        # Get current color source and format it for filename
        color_source = gui_color_mode.value
        color_source_str = color_source.replace(" ", "_").lower()
        
        file_dir = os.path.join(current_scene_dir, f"pts3d_{setting}.npy")
        current_save_dict = np.load(file_dir, allow_pickle=True).item()
        
        if 'seq_name' in current_save_dict and 'frame_str' in current_save_dict:
            seq_name = current_save_dict['seq_name']
            frame_str = current_save_dict['frame_str']
            filename = f"{seq_name}_{frame_str}_{color_source_str}.png"
        else:
            filename = f"{current_scene}_{setting}_{color_source_str}.png"
        
        filepath = os.path.join(save_output_dir, filename)
        
        # Request a render from the client that triggered the button click
        # This captures what the user is currently seeing in the 3D viewer
        render = event.client.camera.get_render(height=1080, width=1920)
        
        # Save the rendered image
        cv2.imwrite(filepath, cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
        
        print(f"Saved 3D view screenshot to: {filepath}")
        print(f"Resolution: {render.shape[1]}x{render.shape[0]}")

    # Initial camera pose.
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0, -70, 0)
        client.camera.look_at = (0, 0, 0)


    print("Point cloud visualization loaded!")
    print("Use the Viser URL shown above to open the viewer.")

    def load_scene_data(new_scene, new_setting=None):
        nonlocal current_scene_dir, available_settings, setting, current_scene
        nonlocal pts3d, colors, conf, conf_colors, focal, c2w, H, W, semantic_colors, height_colors

        current_scene = new_scene
        current_scene_dir = scene_dirs[current_scene]
        available_settings = get_available_settings(current_scene_dir)
        next_setting = new_setting if new_setting in available_settings else choose_default_setting(available_settings)
        gui_setting_dropdown.options = available_settings
        gui_setting_dropdown.value = next_setting
        setting = next_setting

        save_dict = load_data(current_scene_dir, setting)
        pts3d = save_dict['pts3d']
        colors = save_dict['colors']
        conf = save_dict['conf']
        conf_colors = compute_confidence_colors(conf)
        
        # Compute height-based colors using current slider values
        height_colors = compute_height_colors(
            pts3d, 
            y_min_clip=gui_height_min.value, 
            y_max_clip=gui_height_max.value, 
            colormap='JET', 
            invert_y=True
        )
        
        # Load semantic data if available
        if 'semantic_2ds' in save_dict:
            semantic_colors = compute_semantic_colors(save_dict['semantic_2ds'], current_scene_dir)
        else:
            semantic_colors = None
        
        refresh_conf_slider_bounds()
        focal = save_dict['focal']
        c2w = save_dict['c2w']
        H, W = pts3d.shape[1], pts3d.shape[2]

    while True:
        target_scene = gui_scene_dropdown.value
        if target_scene != current_scene:
            load_scene_data(target_scene, gui_setting_dropdown.value)
            need_redraw = True

        target_setting = gui_setting_dropdown.value
        if target_setting != setting:
            load_scene_data(current_scene, target_setting)
            need_redraw = True
        
        if need_redraw:
            # Draw once per change.
            color_source = gui_color_mode.value
            if color_source == "Confidence":
                colors_to_draw = conf_colors
            elif color_source == "Semantic":
                if semantic_colors is not None:
                    colors_to_draw = semantic_colors
                else:
                    print("Warning: Semantic colors not available, using saved colors instead.")
                    colors_to_draw = colors
            elif color_source == "Height":
                colors_to_draw = height_colors
            else:
                colors_to_draw = colors
            
            draw_scene(
                server,
                pts3d,
                colors_to_draw,
                conf,
                gui_conf_threshold.value,
                c2w,
                focal,
                H,
                W,
                scene_handles,
                gui_point_size,
            )
            need_redraw = False
        time.sleep(0.01)

if __name__ == "__main__":
    main()
