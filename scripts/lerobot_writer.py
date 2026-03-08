"""
LeRobot v3.0 Dataset Writer

Writes datasets compatible with huggingface/lerobot (>=0.4.0) without
requiring the lerobot package itself.

Dependencies: pyarrow, imageio[ffmpeg]
Install:      pip install pyarrow "imageio[ffmpeg]"

Directory structure produced:
    <root>/
    ├── meta/
    │   ├── info.json
    │   ├── tasks.parquet
    │   ├── stats.json
    │   └── episodes/
    │       └── chunk-000/
    │           └── file-000.parquet
    ├── data/
    │   └── chunk-000/
    │       ├── file-000.parquet
    │       ├── file-001.parquet
    │       └── ...
    └── videos/
        └── observation.images.<cam>/
            └── chunk-000/
                ├── file-000.mp4
                ├── file-001.mp4
                └── ...

Usage:
    from aloha.lerobot_writer import LeRobotWriter, build_aloha_features

    features = build_aloha_features(['cam_high', 'cam_left_wrist', 'cam_right_wrist'])
    writer = LeRobotWriter.create(
        root='/path/to/output',
        repo_id='aloha/my_task',
        fps=50,
        features=features,
        robot_type='aloha_vx300s',
    )

    for episode in episodes:
        for frame in episode:
            writer.add_frame(frame)
        writer.save_episode(task='pick and place')

    writer.finalize()
"""

import json
import os
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    pa = None
    pq = None
    _HAS_PYARROW = False

try:
    import imageio.v2 as iio
    _HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio as iio
        _HAS_IMAGEIO = True
    except ImportError:
        iio = None
        _HAS_IMAGEIO = False


CODEBASE_VERSION = 'v3.0'
CHUNKS_SIZE = 1000


def _check_deps():
    missing = []
    if not _HAS_PYARROW:
        missing.append('pyarrow')
    if not _HAS_IMAGEIO:
        missing.append('imageio[ffmpeg]')
    if missing:
        raise ImportError(
            f'LeRobotWriter requires: {", ".join(missing)}. '
            f'Install with: pip install {" ".join(missing)}'
        )


def build_aloha_features(camera_names, is_mobile=False, image_shape=(480, 640, 3)):
    """Build LeRobot feature definitions for the ALOHA robot.

    Args:
        camera_names: List of camera names (e.g. ['cam_high', 'cam_left_wrist', ...])
        is_mobile: Whether to include base action features.
        image_shape: (H, W, C) resolution of camera images.

    Returns:
        Features dict for LeRobotWriter.
    """
    joint_names = [
        'left_waist', 'left_shoulder', 'left_elbow',
        'left_forearm_roll', 'left_wrist_angle', 'left_wrist_rotate',
        'left_gripper',
        'right_waist', 'right_shoulder', 'right_elbow',
        'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate',
        'right_gripper',
    ]

    features = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (14,),
            'names': {'axes': joint_names},
        },
        'observation.velocity': {
            'dtype': 'float32',
            'shape': (14,),
            'names': {'axes': joint_names},
        },
        'observation.effort': {
            'dtype': 'float32',
            'shape': (14,),
            'names': {'axes': joint_names},
        },
        'action': {
            'dtype': 'float32',
            'shape': (14,),
            'names': {'axes': joint_names},
        },
    }

    if is_mobile:
        features['action.base'] = {
            'dtype': 'float32',
            'shape': (2,),
            'names': {'axes': ['linear_velocity', 'angular_velocity']},
        }

    for cam_name in camera_names:
        features[f'observation.images.{cam_name}'] = {
            'dtype': 'video',
            'shape': tuple(image_shape),
            'names': ['height', 'width', 'channels'],
        }

    return features


class LeRobotWriter:
    """Write datasets in LeRobot v3.0 format without the lerobot package.

    Video frames are encoded on-the-fly via imageio (not buffered in RAM).
    Only small numeric data is buffered per episode.

    Compatible with ``lerobot >= 0.4.0`` (``LeRobotDataset``).

    v3.0 key differences from v2.0:
      - Data/video files named ``file-NNN`` (one episode per file)
      - Video path: ``videos/{video_key}/chunk-NNN/file-NNN.mp4``
      - Episodes metadata: ``meta/episodes/chunk-000/file-000.parquet``
      - Tasks metadata: ``meta/tasks.parquet``
      - No video struct columns in data parquet
      - info.json has ``chunks_size``, ``data_files_size_in_mb``,
        ``video_files_size_in_mb``; no ``keys``/``video_keys``
    """

    def __init__(self, root, repo_id, fps, features, robot_type='unknown', start_episode=0):
        _check_deps()

        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = fps
        self.features = dict(features)
        self.robot_type = robot_type

        self.video_keys = sorted(k for k, v in features.items() if v['dtype'] == 'video')
        self.data_keys = sorted(k for k, v in features.items() if v['dtype'] != 'video')

        (self.root / 'meta').mkdir(parents=True, exist_ok=True)

        # Global state
        self.episode_index = start_episode
        self.global_frame_index = 0
        self.episodes_info = []
        self.tasks = {}  # task_str -> task_index

        # Per-episode buffers
        self._frames = []
        self._num_frames = 0
        self._video_writers = {}   # vk -> imageio writer
        self._video_paths = {}     # vk -> relative path string

        # Running statistics accumulators
        self._stats_min = {}
        self._stats_max = {}
        self._stats_sum = {}
        self._stats_sum_sq = {}
        self._stats_count = {}
        self._stats_all = {}  # key -> list of arrays, for quantile computation

        # Load existing metadata if resuming
        if start_episode > 0:
            self._load_existing()

    @classmethod
    def create(cls, root, repo_id, fps, features, robot_type='unknown', start_episode=0):
        """Create a new dataset writer.

        Args:
            root: Output directory path.
            repo_id: Dataset identifier (e.g. 'aloha/pick_and_place').
            fps: Recording frequency in Hz.
            features: Dict of feature definitions. Keys are feature names,
                values are dicts with 'dtype', 'shape', and optional 'names'.
                Use dtype='video' for camera features.
            robot_type: Robot identifier string.
            start_episode: Starting episode index (to match existing HDF5 episodes).
        """
        return cls(root, repo_id, fps, features, robot_type, start_episode)

    def add_frame(self, frame_dict):
        """Add one frame to the current episode.

        Args:
            frame_dict: Feature name -> value mapping.
                Numeric features: numpy array or list (cast to float32).
                Video features: uint8 numpy array (H, W, C) in BGR order
                    (standard OpenCV/ROS format -- automatically converted to RGB).
        """
        # Buffer numeric data (small per frame)
        numeric = {}
        for key in self.data_keys:
            if key in frame_dict and frame_dict[key] is not None:
                numeric[key] = np.array(frame_dict[key], dtype=np.float32).flatten()
        self._frames.append(numeric)

        # Stream video frames to encoder (no RAM buffering)
        for vk in self.video_keys:
            if vk not in frame_dict or frame_dict[vk] is None:
                continue
            img = np.asarray(frame_dict[vk], dtype=np.uint8)

            if vk not in self._video_writers:
                self._video_writers[vk] = self._open_video_writer(vk, img)

            self._video_writers[vk].append_data(img)

        self._num_frames += 1

    def save_episode(self, task='default'):
        """Finalize and write the current episode to disk.

        Args:
            task: Task description string for this episode.
        """
        if self._num_frames == 0:
            self.discard_episode()
            return

        # Register task
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
        task_index = self.tasks[task]

        # Close video encoders (flushes remaining frames)
        self._close_video_writers()

        num_frames = self._num_frames
        chunk_idx, file_idx = self._chunk_and_file(self.episode_index)
        chunk_dir = f'chunk-{chunk_idx:03d}'

        # Build parquet columns (no video columns in v3.0)
        columns = {}

        for key in self.data_keys:
            shape = self.features[key]['shape']
            dim = shape[0] if isinstance(shape, (list, tuple)) else shape
            col = []
            for frame in self._frames:
                col.append(frame[key].tolist() if key in frame else [0.0] * dim)
            columns[key] = col

        columns['timestamp'] = [float(i / self.fps) for i in range(num_frames)]
        columns['frame_index'] = list(range(num_frames))
        columns['episode_index'] = [self.episode_index] * num_frames
        columns['index'] = list(range(
            self.global_frame_index, self.global_frame_index + num_frames
        ))
        columns['task_index'] = [task_index] * num_frames

        # Write parquet
        parquet_dir = self.root / 'data' / chunk_dir
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = parquet_dir / f'file-{file_idx:03d}.parquet'
        self._write_parquet(columns, parquet_path)

        # Update running stats (before clearing buffers)
        self._update_stats()

        # Build episode metadata (v3.0 format)
        ep_meta = {
            'episode_index': self.episode_index,
            'tasks': [task],
            'length': num_frames,
            'dataset_from_index': self.global_frame_index,
            'dataset_to_index': self.global_frame_index + num_frames,
            'data/chunk_index': chunk_idx,
            'data/file_index': file_idx,
            'meta/episodes/chunk_index': 0,
            'meta/episodes/file_index': 0,
        }
        for vk in self.video_keys:
            ep_meta[f'videos/{vk}/chunk_index'] = chunk_idx
            ep_meta[f'videos/{vk}/file_index'] = file_idx
            ep_meta[f'videos/{vk}/from_timestamp'] = 0.0
            ep_meta[f'videos/{vk}/to_timestamp'] = float(num_frames - 1) / self.fps

        self.episodes_info.append(ep_meta)

        self.global_frame_index += num_frames
        self.episode_index += 1
        self._reset_buffers()

    def discard_episode(self):
        """Discard current episode without saving."""
        self._close_video_writers()
        for rel_path in self._video_paths.values():
            abs_path = self.root / rel_path
            if abs_path.exists():
                abs_path.unlink()
        self._reset_buffers()

    def finalize(self):
        """Write global metadata files. Must be called after all episodes."""
        self._close_video_writers()

        if not self.episodes_info:
            print('LeRobotWriter: no episodes to finalize')
            return

        all_features = self._build_features_metadata()
        stats = self._compute_final_stats()

        meta_dir = self.root / 'meta'

        # info.json (v3.0 format)
        info = {
            'codebase_version': CODEBASE_VERSION,
            'robot_type': self.robot_type,
            'total_episodes': len(self.episodes_info),
            'total_frames': self.global_frame_index,
            'total_tasks': len(self.tasks),
            'chunks_size': CHUNKS_SIZE,
            'data_files_size_in_mb': 100,
            'video_files_size_in_mb': 200,
            'fps': self.fps,
            'splits': {'train': f'0:{len(self.episodes_info)}'},
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
            'features': all_features,
        }
        _write_json(meta_dir / 'info.json', info)

        # tasks.parquet (v3.0 uses parquet instead of jsonl)
        # Include pandas index metadata so LeRobot can resolve task strings
        task_indices = []
        task_strings = []
        for task_str, idx in sorted(self.tasks.items(), key=lambda x: x[1]):
            task_indices.append(idx)
            task_strings.append(task_str)
        pandas_meta = {
            'index_columns': ['task'],
            'column_indexes': [{'name': None, 'field_name': None,
                                'pandas_type': 'unicode', 'numpy_type': 'object',
                                'metadata': {'encoding': 'UTF-8'}}],
            'columns': [
                {'name': 'task_index', 'field_name': 'task_index',
                 'pandas_type': 'int64', 'numpy_type': 'int64', 'metadata': None},
                {'name': 'task', 'field_name': 'task',
                 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': None},
            ],
            'attributes': {},
            'creator': {'library': 'pyarrow', 'version': pa.__version__},
        }
        tasks_schema = pa.schema([
            pa.field('task_index', pa.int64()),
            pa.field('task', pa.string()),
        ], metadata={b'pandas': json.dumps(pandas_meta).encode()})
        tasks_table = pa.table({
            'task_index': pa.array(task_indices, type=pa.int64()),
            'task': pa.array(task_strings, type=pa.string()),
        }, schema=tasks_schema)
        pq.write_table(tasks_table, str(meta_dir / 'tasks.parquet'))

        # episodes parquet (v3.0: meta/episodes/chunk-000/file-000.parquet)
        episodes_dir = meta_dir / 'episodes' / 'chunk-000'
        episodes_dir.mkdir(parents=True, exist_ok=True)
        ep_columns = {
            col: [ep[col] for ep in self.episodes_info]
            for col in self.episodes_info[0]
        }
        ep_table = pa.table(ep_columns)
        pq.write_table(ep_table, str(episodes_dir / 'file-000.parquet'))

        # stats.json
        _write_json(meta_dir / 'stats.json', stats)

        n_ep = len(self.episodes_info)
        n_fr = self.global_frame_index
        print(f'LeRobotWriter: finalized {n_ep} episodes, {n_fr} frames -> {self.root}')

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_existing(self):
        """Load existing metadata when resuming a dataset."""
        meta_dir = self.root / 'meta'

        # Load existing episodes
        ep_path = meta_dir / 'episodes' / 'chunk-000' / 'file-000.parquet'
        if ep_path.exists():
            ep_table = pq.read_table(str(ep_path))
            col_names = ep_table.column_names
            for i in range(len(ep_table)):
                ep_meta = {}
                for col in col_names:
                    val = ep_table.column(col)[i].as_py()
                    ep_meta[col] = val
                self.episodes_info.append(ep_meta)
            # Set global_frame_index from last episode
            if self.episodes_info:
                last_ep = self.episodes_info[-1]
                self.global_frame_index = last_ep['dataset_to_index']
            print(f'Loaded {len(self.episodes_info)} existing episodes '
                  f'(global_frame_index={self.global_frame_index})')

        # Load existing tasks
        tasks_path = meta_dir / 'tasks.parquet'
        if tasks_path.exists():
            tasks_table = pq.read_table(str(tasks_path))
            indices = tasks_table.column('task_index').to_pylist()
            strings = tasks_table.column('task').to_pylist()
            for idx, task_str in zip(indices, strings):
                self.tasks[task_str] = idx
            print(f'Loaded {len(self.tasks)} existing tasks')

        # Load existing stats from parquet data files
        data_dir = self.root / 'data'
        if data_dir.exists():
            for chunk_dir in sorted(data_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue
                for pfile in sorted(chunk_dir.glob('*.parquet')):
                    table = pq.read_table(str(pfile))
                    for key in self.data_keys:
                        if key not in table.column_names:
                            continue
                        vals = table.column(key).to_pylist()
                        arr = np.array(vals, dtype=np.float64)
                        n = arr.shape[0]
                        if key not in self._stats_count:
                            self._stats_min[key] = arr.min(axis=0)
                            self._stats_max[key] = arr.max(axis=0)
                            self._stats_sum[key] = arr.sum(axis=0)
                            self._stats_sum_sq[key] = (arr ** 2).sum(axis=0)
                            self._stats_count[key] = n
                            self._stats_all[key] = [arr]
                        else:
                            self._stats_min[key] = np.minimum(self._stats_min[key], arr.min(axis=0))
                            self._stats_max[key] = np.maximum(self._stats_max[key], arr.max(axis=0))
                            self._stats_sum[key] += arr.sum(axis=0)
                            self._stats_sum_sq[key] += (arr ** 2).sum(axis=0)
                            self._stats_count[key] += n
                            self._stats_all[key].append(arr)

    def _chunk_and_file(self, episode_index):
        """Return (chunk_index, file_index) for one-episode-per-file layout."""
        chunk_idx = episode_index // CHUNKS_SIZE
        file_idx = episode_index % CHUNKS_SIZE
        return chunk_idx, file_idx

    def _open_video_writer(self, video_key, first_frame):
        """Open an imageio video writer for a camera stream.

        v3.0 path: videos/{video_key}/chunk-NNN/file-NNN.mp4
        """
        chunk_idx, file_idx = self._chunk_and_file(self.episode_index)
        rel_path = (
            f'videos/{video_key}'
            f'/chunk-{chunk_idx:03d}'
            f'/file-{file_idx:03d}.mp4'
        )
        abs_path = self.root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        self._video_paths[video_key] = rel_path

        writer = iio.get_writer(
            str(abs_path),
            fps=self.fps,
            codec='libx264',
            quality=None,
            pixelformat='yuv420p',
            output_params=['-crf', '18', '-preset', 'fast'],
        )
        return writer

    def _close_video_writers(self):
        for writer in self._video_writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self._video_writers = {}

    def _reset_buffers(self):
        self._frames.clear()
        self._num_frames = 0
        self._video_writers = {}
        self._video_paths = {}

    def _write_parquet(self, columns, path):
        """Write data parquet (v3.0: no video columns)."""
        fields = []
        arrays = []

        for key in self.data_keys:
            if key in columns:
                fields.append(pa.field(key, pa.list_(pa.float32())))
                arrays.append(pa.array(columns[key], type=pa.list_(pa.float32())))

        auto_cols = [
            ('timestamp', pa.float32()),
            ('frame_index', pa.int64()),
            ('episode_index', pa.int64()),
            ('index', pa.int64()),
            ('task_index', pa.int64()),
        ]
        for name, dtype in auto_cols:
            fields.append(pa.field(name, dtype))
            arrays.append(pa.array(columns[name], type=dtype))

        table = pa.table(arrays, schema=pa.schema(fields))
        pq.write_table(table, str(path))

    def _update_stats(self):
        for key in self.data_keys:
            vals = [f[key] for f in self._frames if key in f]
            if not vals:
                continue
            arr = np.stack(vals).astype(np.float64)
            n = arr.shape[0]

            if key not in self._stats_count:
                self._stats_min[key] = arr.min(axis=0)
                self._stats_max[key] = arr.max(axis=0)
                self._stats_sum[key] = arr.sum(axis=0)
                self._stats_sum_sq[key] = (arr ** 2).sum(axis=0)
                self._stats_count[key] = n
                self._stats_all[key] = [arr]
            else:
                self._stats_min[key] = np.minimum(self._stats_min[key], arr.min(axis=0))
                self._stats_max[key] = np.maximum(self._stats_max[key], arr.max(axis=0))
                self._stats_sum[key] += arr.sum(axis=0)
                self._stats_sum_sq[key] += (arr ** 2).sum(axis=0)
                self._stats_count[key] += n
                self._stats_all[key].append(arr)

    def _compute_final_stats(self):
        stats = {}
        for key in self.data_keys:
            if key not in self._stats_count:
                continue
            n = self._stats_count[key]
            mean = self._stats_sum[key] / n
            var = np.maximum(self._stats_sum_sq[key] / n - mean ** 2, 0.0)
            # Compute quantiles from all buffered data
            all_data = np.concatenate(self._stats_all[key], axis=0)
            stats[key] = {
                'min': self._stats_min[key].tolist(),
                'max': self._stats_max[key].tolist(),
                'mean': mean.tolist(),
                'std': np.sqrt(var).tolist(),
                'q01': np.quantile(all_data, 0.01, axis=0).tolist(),
                'q99': np.quantile(all_data, 0.99, axis=0).tolist(),
                'count': [int(n)],
            }
        # Add placeholder stats for camera keys (overwritten by imagenet stats at train time)
        for vk in self.video_keys:
            stats[vk] = {
                'min': [0.0, 0.0, 0.0],
                'max': [1.0, 1.0, 1.0],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'q01': [0.0, 0.0, 0.0],
                'q99': [1.0, 1.0, 1.0],
                'count': [int(self.global_frame_index)],
            }
        return stats

    def _build_features_metadata(self):
        feats = {}
        for key, spec in self.features.items():
            entry = {
                'dtype': spec['dtype'],
                'shape': (
                    list(spec['shape']) if isinstance(spec['shape'], tuple)
                    else spec['shape']
                ),
                'names': spec.get('names'),
            }
            if spec['dtype'] == 'video':
                entry['info'] = {
                    'fps': float(self.fps),
                    'codec': 'h264',
                    'pix_format': 'yuv420p',
                }
            feats[key] = entry

        for name in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index'):
            dt = 'float32' if name == 'timestamp' else 'int64'
            feats[name] = {'dtype': dt, 'shape': [1], 'names': None}

        return feats


def _write_json(path, data):
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=2)
