"""YouTube Playlist Streaming — Buffered Producer-Consumer for JEPA Training.

Downloads videos just-in-time from a YouTube playlist via yt-dlp into a temp
directory, then reads frames with OpenCV. A background thread pre-fetches the
next video so the GPU never idles waiting for network I/O.

Usage:
    dataset = YouTubeStreamDataset(
        playlist_url="https://www.youtube.com/playlist?list=PLxxx",
        image_size=224,
        frame_skip=5,
    )
    loader = DataLoader(dataset, batch_size=8, num_workers=0)
    for batch in loader:
        # batch keys: image, image_next, tokens, is_episode_start
        ...
"""

import os
import queue
import shutil
import tempfile
import threading
import time
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


class VideoBuffer:
    """Background thread that downloads playlist videos to a temp directory.

    Producer-consumer pattern:
      - Producer thread: downloads videos one-at-a-time via yt-dlp (480p mp4)
      - Consumer (training loop): reads from the queue of local file paths
      - Cleanup: consumer deletes each video after it's fully read

    Args:
        video_urls: List of YouTube video URLs (or IDs).
        tmp_dir: Temp directory for downloaded files. Created if absent.
        max_resolution: Max video height for yt-dlp format selection.
        prefetch: How many videos to buffer ahead (queue maxsize).
    """

    def __init__(
        self,
        video_urls: list[str],
        tmp_dir: str | None = None,
        max_resolution: int = 480,
        prefetch: int = 2,
    ):
        if yt_dlp is None:
            raise ImportError("yt-dlp is required: pip install yt-dlp")

        self.video_urls = list(video_urls)
        self.max_resolution = max_resolution
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

        # Start producer thread
        self._thread = threading.Thread(target=self._download_loop, daemon=True)
        self._thread.start()

    def _download_loop(self):
        """Producer: download videos sequentially, enqueue local paths."""
        for idx, url in enumerate(self.video_urls):
            if self._stop_event.is_set():
                break

            out_path = os.path.join(self._tmp_dir, f"video_{idx:05d}.mp4")
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
                    ydl.download([url])
                # Enqueue the path (blocks if queue is full — backpressure)
                self._queue.put(out_path)
            except Exception as e:
                print(f"[VideoBuffer] Skipping video {idx} ({url}): {e}")
                continue

        # Sentinel: signal end of playlist
        self._queue.put(None)

    def next_video(self, timeout: float = 300.0) -> str | None:
        """Get next downloaded video path. Returns None when playlist is exhausted."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            print("[VideoBuffer] Timeout waiting for next video download")
            return None

    @staticmethod
    def cleanup_video(path: str):
        """Delete a consumed video file."""
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
            # entry['url'] is the video ID or full URL depending on yt-dlp version
            vid_url = entry.get("url") or entry.get("id")
            if vid_url:
                # Ensure full URL
                if not vid_url.startswith("http"):
                    vid_url = f"https://www.youtube.com/watch?v={vid_url}"
                urls.append(vid_url)
        return urls


def _process_frame(frame_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    """Resize, convert BGR→RGB, normalize with ImageNet stats → (3, H, W) tensor."""
    frame = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # HWC uint8 → CHW float32 [0, 1]
    tensor = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
    # ImageNet normalization
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(tensor)


class YouTubeStreamDataset(IterableDataset):
    """Buffered YouTube playlist dataset for Phase 1.5 temporal JEPA training.

    Yields frame-pair dicts matching TemporalFrameDataset's contract:
        {"image": (3,H,W), "image_next": (3,H,W), "tokens": (L,), "is_episode_start": bool}

    Videos are downloaded just-in-time by a background thread (VideoBuffer).
    Frames are subsampled with frame_skip to expand temporal context.

    Args:
        playlist_url: YouTube playlist URL.
        image_size: Output frame resolution (square).
        frame_skip: Keep every Nth frame (default 5 → ~6 effective FPS from 30fps source).
        max_seq_len: Length of dummy token sequences.
        vocab_size: Vocab size for dummy tokens.
        max_resolution: Max video height for download (default 480).
        tmp_dir: Override temp directory for downloads.
        prefetch: Number of videos to buffer ahead.
    """

    def __init__(
        self,
        playlist_url: str,
        image_size: int = 224,
        frame_skip: int = 5,
        max_seq_len: int = 64,
        vocab_size: int = 32000,
        max_resolution: int = 480,
        tmp_dir: str | None = None,
        prefetch: int = 2,
    ):
        super().__init__()
        self.playlist_url = playlist_url
        self.image_size = image_size
        self.frame_skip = max(frame_skip, 1)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.max_resolution = max_resolution
        self.tmp_dir = tmp_dir
        self.prefetch = prefetch

        # Resolve playlist → list of video URLs (fast, no download)
        print(f"[YouTubeStream] Resolving playlist: {playlist_url}")
        self.video_urls = _get_playlist_video_urls(playlist_url)
        print(f"[YouTubeStream] Found {len(self.video_urls)} videos")

        if not self.video_urls:
            raise ValueError(f"No videos found in playlist: {playlist_url}")

    def _frame_pair_generator(self):
        """Yield (frame_t, frame_t+1) dicts from buffered video files."""
        # Determine which videos this worker handles
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            chunks = np.array_split(self.video_urls, worker_info.num_workers)
            my_urls = list(chunks[worker_info.id])
            worker_tag = f"w{worker_info.id}"
        else:
            my_urls = self.video_urls
            worker_tag = "w0"

        if not my_urls:
            return

        # Create a VideoBuffer for this worker's shard
        worker_tmp = None
        if self.tmp_dir:
            worker_tmp = os.path.join(self.tmp_dir, worker_tag)

        buffer = VideoBuffer(
            video_urls=my_urls,
            tmp_dir=worker_tmp,
            max_resolution=self.max_resolution,
            prefetch=self.prefetch,
        )

        try:
            while True:
                video_path = buffer.next_video(timeout=300.0)
                if video_path is None:
                    # Playlist exhausted
                    break

                yield from self._read_video_pairs(video_path)
                buffer.cleanup_video(video_path)

        finally:
            buffer.stop()

    def _read_video_pairs(self, video_path: str):
        """Read a local video file and yield frame-pair dicts with frame_skip."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[YouTubeStream] Failed to open: {video_path}")
            return

        try:
            prev_frame = None
            frame_count = 0
            is_first_pair = True

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # Apply frame skip: keep every Nth frame
                frame_count += 1
                if (frame_count - 1) % self.frame_skip != 0:
                    continue

                current = _process_frame(frame_bgr, self.image_size)

                if prev_frame is not None:
                    yield {
                        "image": prev_frame,
                        "image_next": current,
                        "tokens": torch.randint(0, self.vocab_size, (self.max_seq_len,)),
                        "is_episode_start": is_first_pair,
                    }
                    is_first_pair = False

                prev_frame = current

        except Exception as e:
            print(f"[YouTubeStream] Error reading {video_path}: {e}")
        finally:
            cap.release()

    def __iter__(self):
        return self._frame_pair_generator()
