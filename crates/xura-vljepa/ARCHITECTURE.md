# xura-vljepa Architecture

## Memory System — Design Decisions

The VisualMemory system makes three deliberate architectural tradeoffs that
increase complexity in exchange for agent continuity. This document explains
why each is intentional and how risks are mitigated.

### 1. Mandatory Memory (not opt-in)

**Decision**: Memory is always attempted on agent instantiation. There is no
`enabled: false` toggle.

**Why**: A World Model is defined by its ability to predict `t+1` based on
`t-1`. Without memory, the agent lives in an "Eternal Now" — it reacts to the
current frame but has no concept of how it got there. This creates a "Groundhog
Day" agent that repeats the same mistakes because it forgot the previous
attempts. Making memory opt-in is equivalent to shipping a lobotomized agent
by default.

**Failure mode**: If the filesystem is unavailable (e.g. read-only container,
permissions error), the agent degrades gracefully — it logs a loud
`DEGRADED MODE` warning and continues without persistence. It does not crash.

```
[Xura] WARNING: failed to open VisualMemory: <error>
[Xura] WARNING: agent running in DEGRADED MODE — no memory persistence!
[Xura] WARNING: the agent will forget everything between sessions.
```

### 2. ffmpeg as External Process

**Decision**: Video recording is done by piping raw RGB frames to an ffmpeg
child process via stdin, rather than using a Rust video encoding library.

**Why**: ffmpeg is the industry standard for video encoding, supports hardware
acceleration (NVENC, VAAPI, VideoToolbox), and produces universally-playable
MP4 files. A Rust-native encoder would add 50k+ lines of dependency for worse
codec support. The agent's video stream is a *log*, not a real-time stream —
ffmpeg handles this use case with zero configuration.

**Complexity managed by**:
- `stderr` is captured via pipe (not suppressed)
- `shutdown()` checks the exit code and returns ffmpeg's error output on failure
- If ffmpeg is not installed, `open()` returns an `io::Error` and the agent
  enters degraded mode (index-only, no video)
- Video is optional per-session (`enable_video: bool` parameter) — text-only
  agents never spawn ffmpeg

### 3. Filesystem Touched on Every Instantiation

**Decision**: `Mamba3Agent::new()` creates the output directory and opens the
index file immediately, rather than deferring to the first `step()` call.

**Why**: Fail-fast. If the filesystem is broken, we want to know at startup —
not 10,000 steps into a training run when the first `remember()` call silently
fails. Early detection means the `DEGRADED MODE` warning appears in the first
line of logs, not buried hours later.

**Costs mitigated by**:
- The operation is a single `create_dir_all` + `File::create` — microseconds
- Retention cleanup (`enforce_retention`) runs on open, preventing unbounded
  disk growth across sessions
- All files are local-only (`~/.xura/memory`), never uploaded or transmitted
- Transparent startup log shows exactly what's happening:

```
[Xura] Memory index: /home/user/.xura/memory/memory.index (Local Only, 7-day rolling window)
```

## Data Lifecycle

```
Agent::new()
  └─ VisualMemory::open()
       ├─ create_dir_all(~/.xura/memory)
       ├─ enforce_retention()          ← delete expired/oversized files
       ├─ File::create(memory.index)   ← index stream ready
       └─ [optional] spawn ffmpeg      ← video stream ready

Agent::step()  ×N
  └─ remember()
       ├─ append to memory.index       ← u64 length-prefixed bincode
       └─ write RGB frame to ffmpeg    ← if video enabled

Agent::drop()
  └─ shutdown()
       ├─ flush index
       ├─ close ffmpeg stdin (→ MP4 finalized)
       └─ wait() + check exit code
```

## Retention Policy

Files are automatically cleaned on session open:

| Setting              | Default | Description                            |
|----------------------|---------|----------------------------------------|
| `max_retention_days` | 7       | Delete rotated files older than N days |
| `max_storage_bytes`  | 10 GB   | Trim oldest files if over budget       |

Only Xura's own rotated files are managed (`memory_*.index`, `history_*.mp4`).
Unrelated files in the output directory are never touched.

## Privacy Guarantee

All memory data stays **local only** — it is never uploaded, transmitted, or
shared. The output directory defaults to `~/.xura/memory` on the user's machine.
