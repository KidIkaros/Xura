"""Video Streaming — Source-Agnostic Buffered Producer-Consumer for JEPA Training.

Supports any video source: YouTube playlists, local directories, HTTP/S3 URLs.
Videos are acquired just-in-time by a background thread, then read with OpenCV
and chunked into (T, C, H, W) frame sequences for Mamba's parallel scan.

Usage:
    # YouTube playlist
    source = YouTubeSource("https://www.youtube.com/playlist?list=PLxxx")

    # Local directory of video files
    source = LocalDirectorySource("/data/videos")

    # Arbitrary HTTP/S3 URLs
    source = URLListSource(["https://cdn.example.com/v1.mp4", ...])

    dataset = VideoStreamDataset(source, image_size=224, seq_len=16, frame_skip=5)
    loader = DataLoader(dataset, batch_size=8, num_workers=0)
    for batch in loader:
        # batch["frames"]: (B, T, C, H, W) — T-frame chunks for Mamba parallel scan
        # batch["tokens"]: (B, L) — dummy tokens
        # batch["is_episode_start"]: (B,) — True on first chunk of each video
        ...
"""

import atexit
import os
import queue
import shutil
import tempfile
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# ImageNet normalization constants (match TemporalFrameDataset)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}
_MIN_VIDEO_FILE_BYTES = 10_000  # Minimum file size to consider a download valid


# ═══════════════════════════════════════════════════════════════════════════
# Video Sources — pluggable backends for acquiring video files
# ═══════════════════════════════════════════════════════════════════════════

class VideoSource(ABC):
    """Abstract base class for video acquisition backends.

    A VideoSource provides an ordered list of video identifiers and knows
    how to make each one available as a local file path. The VideoBuffer
    calls acquire() from a background thread to fetch videos just-in-time.

    Subclass contract:
      - video_ids: list of opaque identifiers (URLs, paths, etc.)
      - acquire(video_id, out_dir) → local file path or None on failure
      - needs_cleanup: whether acquired files should be deleted after use
    """

    @property
    @abstractmethod
    def video_ids(self) -> list[str]:
        """Ordered list of video identifiers to process."""
        ...

    @abstractmethod
    def acquire(self, video_id: str, out_dir: str, idx: int) -> str | None:
        """Make a video available as a local file.

        Args:
            video_id: Identifier from video_ids.
            out_dir: Temp directory for downloads (ignored for local files).
            idx: Sequential index for unique filenames.

        Returns:
            Local file path, or None if acquisition failed.
        """
        ...

    @property
    def needs_cleanup(self) -> bool:
        """Whether acquired files should be deleted after reading."""
        return True


class YouTubeSource(VideoSource):
    """Acquire videos from a YouTube playlist via yt-dlp.

    Args:
        playlist_url: YouTube playlist URL.
        max_resolution: Max video height for format selection (default 480).
    """

    def __init__(self, playlist_url: str, max_resolution: int = 480):
        if yt_dlp is None:
            raise ImportError("yt-dlp is required: pip install yt-dlp")
        self.playlist_url = playlist_url
        self.max_resolution = max_resolution
        self._video_urls = _get_playlist_video_urls(playlist_url)
        print(f"[YouTubeSource] Resolved {len(self._video_urls)} videos from playlist")

    @property
    def video_ids(self) -> list[str]:
        return self._video_urls

    def acquire(self, video_id: str, out_dir: str, idx: int) -> str | None:
        out_path = os.path.join(out_dir, f"video_{idx:05d}.mp4")
        ydl_opts = {
            "format": f"best[height<={self.max_resolution}][ext=mp4]/"
                      f"best[height<={self.max_resolution}]/best",
            "outtmpl": out_path,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "socket_timeout": 30,
            "retries": 3,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_id])
            if os.path.isfile(out_path) and os.path.getsize(out_path) >= _MIN_VIDEO_FILE_BYTES:
                return out_path
            if os.path.isfile(out_path):
                os.remove(out_path)
        except Exception as e:
            print(f"[YouTubeSource] Failed to download {video_id}: {e}")
        return None


class LocalDirectorySource(VideoSource):
    """Serve videos from a local directory (no download needed).

    Scans for common video extensions: .mp4, .avi, .mkv, .mov, .webm, .flv, .wmv

    Args:
        video_dir: Path to directory containing video files.
        recursive: Whether to search subdirectories (default False).
    """

    def __init__(self, video_dir: str, recursive: bool = False):
        self.video_dir = Path(video_dir)
        if not self.video_dir.is_dir():
            raise ValueError(f"Video directory not found: {video_dir}")
        pattern = "**/*" if recursive else "*"
        self._paths = sorted([
            str(p) for p in self.video_dir.glob(pattern)
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
        ])
        print(f"[LocalDirectorySource] Found {len(self._paths)} videos in {video_dir}")

    @property
    def video_ids(self) -> list[str]:
        return self._paths

    def acquire(self, video_id: str, out_dir: str, idx: int) -> str | None:
        # Already a local path — just validate it exists
        if os.path.isfile(video_id):
            return video_id
        print(f"[LocalDirectorySource] File not found: {video_id}")
        return None

    @property
    def needs_cleanup(self) -> bool:
        return False  # Don't delete the user's source files


class URLListSource(VideoSource):
    """Download videos from arbitrary HTTP/HTTPS URLs (S3 presigned, CDN, etc.).

    Args:
        urls: List of direct video URLs.
        timeout: Download timeout in seconds per video (default 300).
    """

    def __init__(self, urls: list[str], timeout: int = 300):
        self._urls = list(urls)
        self.timeout = timeout
        print(f"[URLListSource] {len(self._urls)} video URLs")

    @property
    def video_ids(self) -> list[str]:
        return self._urls

    def acquire(self, video_id: str, out_dir: str, idx: int) -> str | None:
        # Infer extension from URL, default to .mp4
        ext = Path(video_id.split("?")[0]).suffix or ".mp4"
        out_path = os.path.join(out_dir, f"video_{idx:05d}{ext}")
        try:
            with urllib.request.urlopen(video_id, timeout=self.timeout) as resp:
                with open(out_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
            if os.path.isfile(out_path) and os.path.getsize(out_path) >= _MIN_VIDEO_FILE_BYTES:
                return out_path
            if os.path.isfile(out_path):
                os.remove(out_path)
            print(f"[URLListSource] Download too small/empty for {video_id}")
        except Exception as e:
            print(f"[URLListSource] Failed to download {video_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# VideoBuffer — background prefetch for any VideoSource
# ═══════════════════════════════════════════════════════════════════════════

class VideoBuffer:
    """Background thread that acquires videos from any VideoSource.

    Producer-consumer pattern:
      - Producer thread: calls source.acquire() sequentially, enqueues local paths
      - Consumer (training loop): reads from the queue of local file paths
      - Cleanup: consumer deletes each video after reading (if source.needs_cleanup)

    Args:
        source: A VideoSource providing video identifiers and acquisition logic.
        video_ids: Subset of source.video_ids for this worker.
        tmp_dir: Temp directory for downloads. Created if absent.
        prefetch: How many videos to buffer ahead (queue maxsize).
    """

    def __init__(
        self,
        source: VideoSource,
        video_ids: list[str],
        tmp_dir: str | None = None,
        prefetch: int = 2,
    ):
        self.source = source
        self.video_ids = list(video_ids)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=prefetch)
        self._stop_event = threading.Event()

        # Temp directory management
        if tmp_dir is not None:
            self._tmp_dir = tmp_dir
            self._owns_tmp = False
            os.makedirs(tmp_dir, exist_ok=True)
        else:
            self._tmp_dir = tempfile.mkdtemp(prefix="xura_buf_")
            self._owns_tmp = True

        # Register atexit cleanup (more reliable than __del__)
        atexit.register(self.stop)

        # Start producer thread
        self._thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._thread.start()

    def _acquire_loop(self):
        """Producer: acquire videos sequentially, enqueue local paths."""
        for idx, vid_id in enumerate(self.video_ids):
            if self._stop_event.is_set():
                break

            path = self.source.acquire(vid_id, self._tmp_dir, idx)
            if path is not None:
                self._queue.put(path)

        # Sentinel: signal end of source
        self._queue.put(None)

    def next_video(self, timeout: float = 300.0) -> str | None:
        """Get next acquired video path. Returns None when source is exhausted."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            print("[VideoBuffer] Timeout waiting for next video")
            return None

    def cleanup_video(self, path: str):
        """Delete a consumed video file (only if source requires cleanup)."""
        if self.source.needs_cleanup:
            try:
                os.remove(path)
            except OSError:
                pass

    def stop(self):
        """Signal the producer to stop and clean up temp directory."""
        self._stop_event.set()
        if self._owns_tmp and os.path.isdir(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def __del__(self):
        # Fallback; atexit handler is the primary cleanup path
        self.stop()


def _get_playlist_video_urls(playlist_url: str) -> list[str]:
    """Extract individual video URLs from a YouTube playlist (no download)."""
    if yt_dlp is None:
        raise ImportError("yt-dlp is required: pip install yt-dlp")

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        entries = info.get("entries", [])
        urls = []
        for entry in entries:
            vid_url = entry.get("url") or entry.get("id")
            if vid_url:
                if not vid_url.startswith("http"):
                    vid_url = f"https://www.youtube.com/watch?v={vid_url}"
                urls.append(vid_url)
        return urls


# ═══════════════════════════════════════════════════════════════════════════
# Frame processing
# ═══════════════════════════════════════════════════════════════════════════

def _process_frame(frame_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    """Resize, convert BGR→RGB, normalize with ImageNet stats → (3, H, W) tensor."""
    frame = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # HWC uint8 → CHW float32 [0, 1]
    tensor = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
    # ImageNet normalization
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(tensor)


# ═══════════════════════════════════════════════════════════════════════════
# VideoStreamDataset — source-agnostic streaming dataset
# ═══════════════════════════════════════════════════════════════════════════

class VideoStreamDataset(IterableDataset):
    """Source-agnostic video streaming dataset for temporal JEPA training.

    Yields frame-chunk dicts matching TemporalFrameDataset's contract:
        {"frames": (T, 3, H, W), "tokens": (L,), "is_episode_start": bool}

    Each chunk contains T contiguous (after frame_skip) frames from a single
    video. Mamba processes all T frames in one parallel scan pass. BPTT state
    carries across sequential chunks.

    Works with any VideoSource: YouTube, local files, HTTP URLs, etc.

    Args:
        source: A VideoSource that provides video file paths.
        image_size: Output frame resolution (square).
        seq_len: Number of frames per chunk (default 16).
        frame_skip: Keep every Nth frame (default 5 → ~6 effective FPS from 30fps source).
        max_seq_len: Length of dummy token sequences.
        vocab_size: Vocab size for dummy tokens.
        tmp_dir: Override temp directory for downloads.
        prefetch: Number of videos to buffer ahead.
    """

    def __init__(
        self,
        source: VideoSource,
        image_size: int = 224,
        seq_len: int = 16,
        frame_skip: int = 5,
        max_seq_len: int = 64,
        vocab_size: int = 32000,
        tmp_dir: str | None = None,
        prefetch: int = 2,
    ):
        super().__init__()
        self.source = source
        self.image_size = image_size
        self.seq_len = seq_len
        self.frame_skip = max(frame_skip, 1)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.tmp_dir = tmp_dir
        self.prefetch = prefetch

        if not source.video_ids:
            raise ValueError("VideoSource has no videos")
        print(f"[VideoStreamDataset] {len(source.video_ids)} videos, seq_len={seq_len}")

    def _chunk_generator(self):
        """Yield (T, C, H, W) frame-chunk dicts from buffered video files."""
        all_ids = self.source.video_ids

        # Determine which videos this worker handles
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            chunks = np.array_split(all_ids, worker_info.num_workers)
            my_ids = list(chunks[worker_info.id])
            worker_tag = f"w{worker_info.id}"
        else:
            my_ids = all_ids
            worker_tag = "w0"

        if not my_ids:
            return

        # Create a VideoBuffer for this worker's shard
        worker_tmp = None
        if self.tmp_dir:
            worker_tmp = os.path.join(self.tmp_dir, worker_tag)

        buffer = VideoBuffer(
            source=self.source,
            video_ids=my_ids,
            tmp_dir=worker_tmp,
            prefetch=self.prefetch,
        )

        try:
            while True:
                video_path = buffer.next_video(timeout=300.0)
                if video_path is None:
                    break

                yield from self._read_video_chunks(video_path)
                buffer.cleanup_video(video_path)

        finally:
            buffer.stop()

    def _read_video_chunks(self, video_path: str):
        """Read a local video file and yield frame-chunk dicts with frame_skip."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[VideoStreamDataset] Failed to open: {video_path}")
            return

        try:
            frame_buffer = []
            frame_count = 0
            is_first_chunk = True

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # Apply frame skip: keep every Nth frame
                frame_count += 1
                if (frame_count - 1) % self.frame_skip != 0:
                    continue

                frame_buffer.append(_process_frame(frame_bgr, self.image_size))

                # Yield a chunk when buffer is full
                if len(frame_buffer) == self.seq_len:
                    yield {
                        "frames": torch.stack(frame_buffer),  # (T, C, H, W)
                        "tokens": torch.randint(0, self.vocab_size, (self.max_seq_len,)),
                        "is_episode_start": is_first_chunk,
                    }
                    is_first_chunk = False
                    frame_buffer = []

        except Exception as e:
            print(f"[VideoStreamDataset] Error reading {video_path}: {e}")
        finally:
            cap.release()

    def __iter__(self):
        return self._chunk_generator()


def YouTubeStreamDataset(
    playlist_url: str,
    image_size: int = 224,
    seq_len: int = 16,
    frame_skip: int = 5,
    max_seq_len: int = 64,
    vocab_size: int = 32000,
    max_resolution: int = 480,
    tmp_dir: str | None = None,
    prefetch: int = 2,
) -> VideoStreamDataset:
    """Convenience constructor: YouTube playlist → VideoStreamDataset."""
    source = YouTubeSource(playlist_url, max_resolution=max_resolution)
    return VideoStreamDataset(
        source=source,
        image_size=image_size,
        seq_len=seq_len,
        frame_skip=frame_skip,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        tmp_dir=tmp_dir,
        prefetch=prefetch,
    )
