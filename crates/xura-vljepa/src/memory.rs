//! VisualMemory — disk-backed dual-stream agent memory.
//!
//! Replaces the ephemeral `Vec<EpisodicEntry>` with two synchronized streams:
//!
//! - **Stream A (Video)**: raw image frames piped to ffmpeg → `history.mp4`
//! - **Stream B (Index)**: bincode-serialized `IndexEntry` → `memory.index`
//!
//! This gives the agent infinite context: it can run for days, generating a
//! highly compressed video log of its life. To "remember" things later, scan
//! the lightweight index file for high-similarity embeddings, look up the
//! timestamp, and seek to that exact second in the video.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::config::VisualMemoryConfig;

// ═══════════════════════════════════════════════════════════════════════════
// Index entry — one per frame, serialized with bincode
// ═══════════════════════════════════════════════════════════════════════════

/// A single entry in the memory index file.
///
/// Each entry corresponds to exactly one video frame (or one text-only step).
/// Embeddings are stored as raw f16 bytes to match the half-precision storage
/// used elsewhere in the agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Agent step number.
    pub step: u64,
    /// Milliseconds since memory session started.
    pub timestamp_ms: u64,
    /// Embedding as raw bytes (f16 values, 2 bytes each).
    pub embedding_bytes: Vec<u8>,
    /// Response tokens produced at this step.
    pub response_tokens: Vec<usize>,
    /// Whether this step had a visual frame.
    pub has_frame: bool,
}

impl IndexEntry {
    /// Create a new index entry, converting f32 embedding to f16 bytes.
    pub fn new(
        step: usize,
        timestamp_ms: u64,
        embedding: &[f32],
        response_tokens: &[usize],
        has_frame: bool,
    ) -> Self {
        let embedding_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();

        Self {
            step: step as u64,
            timestamp_ms,
            embedding_bytes,
            response_tokens: response_tokens.to_vec(),
            has_frame,
        }
    }

    /// Recover the embedding as f32 from stored f16 bytes.
    pub fn embedding_f32(&self) -> Vec<f32> {
        self.embedding_bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bytes = [chunk[0], chunk[1]];
                f16::from_le_bytes(bytes).to_f32()
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VisualMemory — the dual-stream recorder
// ═══════════════════════════════════════════════════════════════════════════

/// Disk-backed dual-stream memory for persistent agent recording.
///
/// Manages:
/// - An ffmpeg child process receiving raw RGB frames via stdin pipe
/// - A bincode index file storing per-frame metadata and embeddings
///
/// Frame count is tracked to ensure synchronization between streams.
pub struct VisualMemory {
    /// Stream A: ffmpeg child process (None if video disabled / text-only).
    ffmpeg_child: Option<Child>,
    /// Stream A: pipe to ffmpeg's stdin.
    video_pipe: Option<std::process::ChildStdin>,
    /// Stream B: buffered writer for the bincode index.
    index_writer: BufWriter<File>,
    /// Number of frames written (sync counter).
    frame_count: u64,
    /// Number of index entries written.
    index_count: u64,
    /// Session start time for timestamp calculation.
    start_time: Instant,
    /// Configuration snapshot.
    config: VisualMemoryConfig,
    /// Whether shutdown has been called.
    is_shutdown: bool,
}

impl VisualMemory {
    /// Open a new VisualMemory session.
    ///
    /// Creates the output directory, opens the index file, and optionally
    /// spawns the ffmpeg process for video recording.
    ///
    /// # Arguments
    /// - `config`: memory configuration
    /// - `enable_video`: whether to spawn ffmpeg (false for text-only agents)
    pub fn open(config: VisualMemoryConfig, enable_video: bool) -> io::Result<Self> {
        // Create output directory
        fs::create_dir_all(&config.output_dir)?;

        // Open index file
        let index_path = format!("{}/memory.index", config.output_dir);
        let index_file = File::create(&index_path)?;
        let index_writer = BufWriter::new(index_file);

        // Spawn ffmpeg if video is enabled
        let (ffmpeg_child, video_pipe) = if enable_video {
            let video_path = format!("{}/history.mp4", config.output_dir);
            let mut child = Command::new(&config.ffmpeg_path)
                .args([
                    "-y",                  // overwrite output
                    "-f", "rawvideo",      // input format
                    "-pix_fmt", "rgb24",   // pixel format
                    "-s", &format!("{}x{}", config.frame_width, config.frame_height),
                    "-r", &config.fps.to_string(),
                    "-i", "pipe:0",        // read from stdin
                    "-c:v", &config.codec,
                    "-crf", &config.crf.to_string(),
                    "-pix_fmt", "yuv420p", // output pixel format (compatibility)
                    &video_path,
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()?;

            let stdin = child.stdin.take();
            (Some(child), stdin)
        } else {
            (None, None)
        };

        Ok(Self {
            ffmpeg_child,
            video_pipe,
            index_writer,
            frame_count: 0,
            index_count: 0,
            start_time: Instant::now(),
            config,
            is_shutdown: false,
        })
    }

    /// Open an index-only session (no ffmpeg, no video).
    ///
    /// Convenience wrapper for text-only agents.
    pub fn open_index_only(config: VisualMemoryConfig) -> io::Result<Self> {
        Self::open(config, false)
    }

    /// Append one step to the memory.
    ///
    /// If `image_rgb` is provided and video is enabled, writes the raw frame
    /// to ffmpeg's stdin. Always writes an index entry.
    ///
    /// # Arguments
    /// - `image_rgb`: optional raw RGB bytes (frame_width * frame_height * 3)
    /// - `embedding`: the agent's embedding for this step
    /// - `response_tokens`: decoded tokens (may be empty)
    /// - `step`: agent step number
    pub fn append(
        &mut self,
        image_rgb: Option<&[u8]>,
        embedding: &[f32],
        response_tokens: &[usize],
        step: usize,
    ) -> io::Result<()> {
        let timestamp_ms = self.start_time.elapsed().as_millis() as u64;
        let has_frame = image_rgb.is_some();

        // Stream A: write video frame
        if let (Some(ref mut pipe), Some(rgb)) = (&mut self.video_pipe, image_rgb) {
            let expected_size = self.config.frame_width * self.config.frame_height * 3;
            if rgb.len() == expected_size {
                pipe.write_all(rgb)?;
                self.frame_count += 1;
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "frame size mismatch: expected {} bytes ({}x{}x3), got {}",
                        expected_size, self.config.frame_width, self.config.frame_height, rgb.len()
                    ),
                ));
            }
        }

        // Stream B: write index entry
        let entry = IndexEntry::new(step, timestamp_ms, embedding, response_tokens, has_frame);
        let encoded = bincode::serialize(&entry).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("bincode serialize: {}", e))
        })?;

        // Write length-prefixed entry (u32 LE length + payload)
        let len = encoded.len() as u32;
        self.index_writer.write_all(&len.to_le_bytes())?;
        self.index_writer.write_all(&encoded)?;
        self.index_count += 1;

        Ok(())
    }

    /// Flush both streams to disk without closing them.
    pub fn flush(&mut self) -> io::Result<()> {
        if let Some(ref mut pipe) = self.video_pipe {
            pipe.flush()?;
        }
        self.index_writer.flush()?;
        Ok(())
    }

    /// Gracefully shut down: flush streams, close ffmpeg pipe, wait for child.
    pub fn shutdown(&mut self) -> io::Result<()> {
        if self.is_shutdown {
            return Ok(());
        }
        self.is_shutdown = true;

        // Flush index
        self.index_writer.flush()?;

        // Close video pipe (signals EOF to ffmpeg → finalizes MP4)
        self.video_pipe.take();

        // Wait for ffmpeg to finish
        if let Some(ref mut child) = self.ffmpeg_child {
            let _ = child.wait();
        }
        self.ffmpeg_child.take();

        Ok(())
    }

    /// Number of video frames written.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Number of index entries written.
    pub fn index_count(&self) -> u64 {
        self.index_count
    }

    /// Whether the video stream is active.
    pub fn has_video(&self) -> bool {
        self.video_pipe.is_some()
    }
}

impl Drop for VisualMemory {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Index reader — for scanning saved memories
// ═══════════════════════════════════════════════════════════════════════════

/// Read all index entries from a memory.index file.
pub fn read_index(path: &str) -> io::Result<Vec<IndexEntry>> {
    use std::io::Read;
    let mut file = File::open(path)?;
    let mut entries = Vec::new();

    loop {
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 10_000_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("index entry too large ({} bytes), likely corrupt", len),
            ));
        }
        let mut payload = vec![0u8; len];
        file.read_exact(&mut payload)?;
        let entry: IndexEntry = bincode::deserialize(&payload).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("bincode deserialize: {}", e))
        })?;
        entries.push(entry);
    }

    Ok(entries)
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility: convert f32 image to raw RGB bytes
// ═══════════════════════════════════════════════════════════════════════════

/// Convert an f32 image (values in [0, 1] or [-1, 1]) to raw RGB u8 bytes.
///
/// Input: flattened (C, H, W) with C=3, values clamped to [0, 255].
/// Output: (H, W, 3) interleaved RGB bytes suitable for ffmpeg rawvideo.
pub fn f32_image_to_rgb_bytes(
    image: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<u8> {
    assert_eq!(channels, 3, "expected 3 channels for RGB");
    assert_eq!(
        image.len(),
        channels * height * width,
        "image size mismatch"
    );

    let mut rgb = vec![0u8; height * width * 3];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                // CHW layout → HWC layout
                let val = image[c * height * width + y * width + x];
                // Clamp and scale: assume [0,1] range; if negative, treat as [-1,1]
                let byte = if val < 0.0 {
                    ((val + 1.0) * 127.5).clamp(0.0, 255.0) as u8
                } else {
                    (val * 255.0).clamp(0.0, 255.0) as u8
                };
                rgb[(y * width + x) * 3 + c] = byte;
            }
        }
    }
    rgb
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_dir(name: &str) -> String {
        let path = env::temp_dir().join(format!("xura_memory_test_{}", name));
        let _ = fs::remove_dir_all(&path);
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn test_index_entry_roundtrip() {
        let embedding = vec![1.0f32, -0.5, 0.0, 3.14];
        let entry = IndexEntry::new(42, 1000, &embedding, &[10, 20, 30], true);

        assert_eq!(entry.step, 42);
        assert_eq!(entry.timestamp_ms, 1000);
        assert!(entry.has_frame);
        assert_eq!(entry.response_tokens, vec![10, 20, 30]);

        // f16 roundtrip: should be close but not exact
        let restored = entry.embedding_f32();
        assert_eq!(restored.len(), embedding.len());
        for (a, b) in embedding.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.01, "f16 roundtrip: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_index_only_append_and_read() {
        let dir = temp_dir("index_only");
        let config = VisualMemoryConfig {
            enabled: true,
            output_dir: dir.clone(),
            ..Default::default()
        };

        // Write entries
        {
            let mut mem = VisualMemory::open_index_only(config).unwrap();
            assert!(!mem.has_video());

            for i in 0..5 {
                let emb = vec![i as f32 * 0.1; 32];
                mem.append(None, &emb, &[i, i + 1], i).unwrap();
            }
            assert_eq!(mem.index_count(), 5);
            mem.shutdown().unwrap();
        }

        // Read them back
        let index_path = format!("{}/memory.index", dir);
        let entries = read_index(&index_path).unwrap();
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].step, 0);
        assert_eq!(entries[4].step, 4);
        assert_eq!(entries[2].response_tokens, vec![2, 3]);
        assert!(!entries[0].has_frame);

        // Clean up
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_frame_count_sync() {
        let dir = temp_dir("frame_sync");
        let config = VisualMemoryConfig {
            enabled: true,
            output_dir: dir.clone(),
            ..Default::default()
        };

        let mut mem = VisualMemory::open_index_only(config).unwrap();

        // Text-only steps: frame_count stays 0, index_count increments
        mem.append(None, &[0.1; 16], &[], 1).unwrap();
        mem.append(None, &[0.2; 16], &[], 2).unwrap();
        assert_eq!(mem.frame_count(), 0);
        assert_eq!(mem.index_count(), 2);

        mem.shutdown().unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_f32_to_rgb_bytes() {
        // Simple 2x2 red image
        let mut image = vec![0.0f32; 3 * 2 * 2];
        // Channel 0 (R) = 1.0 for all pixels
        image[0] = 1.0;
        image[1] = 1.0;
        image[2] = 1.0;
        image[3] = 1.0;

        let rgb = f32_image_to_rgb_bytes(&image, 3, 2, 2);
        assert_eq!(rgb.len(), 2 * 2 * 3);

        // First pixel: R=255, G=0, B=0
        assert_eq!(rgb[0], 255);
        assert_eq!(rgb[1], 0);
        assert_eq!(rgb[2], 0);
    }

    #[test]
    fn test_double_shutdown_safe() {
        let dir = temp_dir("double_shutdown");
        let config = VisualMemoryConfig {
            enabled: true,
            output_dir: dir.clone(),
            ..Default::default()
        };

        let mut mem = VisualMemory::open_index_only(config).unwrap();
        mem.shutdown().unwrap();
        mem.shutdown().unwrap(); // should not panic
        let _ = fs::remove_dir_all(&dir);
    }
}
