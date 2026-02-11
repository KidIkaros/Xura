//! Mamba3 Agent — the full State-Space Delegation (SSD) loop.
//!
//! This is the "real model": a persistent agent that pairs fast Mamba-3
//! perception with slow tool-based reasoning. The loop:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                      Agent.step(input)                          │
//! │                                                                  │
//! │  1. PERCEIVE: encode image/text → Mamba-3 embedding             │
//! │       ↓                                                          │
//! │  2. MONITOR: confusion probe on hidden state                    │
//! │       ↓ confused?                                                │
//! │  3. DELEGATE: select tool → execute → get knowledge             │
//! │       ↓                                                          │
//! │  4. INJECT: h_new = h + W_inject · V_knowledge                  │
//! │       ↓                                                          │
//! │  5. ACT: decode embedding → text response OR tool call          │
//! │       ↓                                                          │
//! │  6. REMEMBER: store (query, response) in episodic memory        │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! The Mamba-3 SSM state is **never cleared** between steps (when
//! `persistent_state = true`), so the agent develops a continuous
//! stream of consciousness that accumulates identity and working style.

use half::f16;

use crate::config::{AgentConfig, Mamba3JepaConfig};
use crate::memory::{VisualMemory, f32_image_to_rgb_bytes};
use crate::recursion::StateInjector;
use crate::selective_decode::SelectiveDecoder;
use crate::tools::{Tool, ToolRegistry, ToolRequest, ToolResult};
use crate::vljepa::{InferenceMode, InferenceOutput, Mamba3Jepa};

// ═══════════════════════════════════════════════════════════════════════════
// VisualGate — skip ViT on static frames (fixes "Retina Burn")
// ═══════════════════════════════════════════════════════════════════════════

/// Caches ViT output and skips re-encoding when the image hasn't changed.
///
/// Uses FNV-1a hashing on sampled pixels for O(1) change detection.
/// Cached visual tokens are stored in f16 to halve memory usage.
pub struct VisualGate {
    last_hash: u64,
    /// Cached ViT visual tokens, stored as f16.
    cached_tokens: Vec<f16>,
    /// Number of patches in the cached output.
    num_patches: usize,
    /// Cache hit count (for diagnostics).
    pub hits: usize,
    /// Cache miss count (for diagnostics).
    pub misses: usize,
}

impl VisualGate {
    pub fn new() -> Self {
        Self {
            last_hash: 0,
            cached_tokens: Vec::new(),
            num_patches: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Fast FNV-1a hash of sampled image pixels.
    fn hash_image(image: &[f32]) -> u64 {
        let n = image.len();
        if n == 0 {
            return 0;
        }
        let step = (n / 256).max(1);
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        let mut i = 0;
        while i < n {
            let clamped = image[i].clamp(-1.0, 1.0);
            let bits = (clamped * 65536.0) as i64 as u64;
            hash ^= bits;
            hash = hash.wrapping_mul(0x0100_0000_01b3); // FNV prime
            i += step;
        }
        hash
    }

    /// Check if image matches cache. Returns cached f32 tokens on hit.
    pub fn check(&mut self, image: &[f32]) -> Option<(Vec<f32>, usize)> {
        let hash = Self::hash_image(image);
        if hash == self.last_hash && !self.cached_tokens.is_empty() {
            self.hits += 1;
            let tokens: Vec<f32> = self.cached_tokens.iter().map(|v| v.to_f32()).collect();
            Some((tokens, self.num_patches))
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store ViT visual tokens in cache (as f16).
    pub fn store(&mut self, image: &[f32], tokens: &[f32], num_patches: usize) {
        self.last_hash = Self::hash_image(image);
        self.cached_tokens = tokens.iter().map(|&v| f16::from_f32(v)).collect();
        self.num_patches = num_patches;
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.last_hash = 0;
        self.cached_tokens.clear();
        self.num_patches = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Agent Input / Output types
// ═══════════════════════════════════════════════════════════════════════════

/// Input to the agent on each step.
#[derive(Clone, Debug)]
pub struct AgentInput {
    /// Visual input (optional): flattened (1, C, H, W).
    pub image: Option<Vec<f32>>,
    /// Image height (required if image is Some).
    pub image_h: usize,
    /// Image width (required if image is Some).
    pub image_w: usize,
    /// Text query as token IDs.
    pub query_tokens: Vec<usize>,
}

impl AgentInput {
    /// Create a text-only input.
    pub fn text(tokens: Vec<usize>) -> Self {
        Self {
            image: None,
            image_h: 0,
            image_w: 0,
            query_tokens: tokens,
        }
    }

    /// Create a vision + text input.
    pub fn vision_text(image: Vec<f32>, h: usize, w: usize, tokens: Vec<usize>) -> Self {
        Self {
            image: Some(image),
            image_h: h,
            image_w: w,
            query_tokens: tokens,
        }
    }
}

/// Output from a single agent step.
#[derive(Clone, Debug)]
pub struct AgentOutput {
    /// The predicted embedding from the JEPA predictor.
    pub embedding: Vec<f32>,
    /// Decoded text tokens (empty if selective decoding suppressed output).
    pub response_tokens: Vec<usize>,
    /// Tool calls made during this step.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Whether the agent decided to decode (vs. staying silent).
    pub did_decode: bool,
    /// Confusion score at the time of this step.
    pub confusion_score: f32,
    /// State drift at the time of this step.
    pub state_drift: f32,
    /// Step number (monotonically increasing).
    pub step_number: usize,
}

/// Record of a tool call made during an agent step.
#[derive(Clone, Debug)]
pub struct ToolCallRecord {
    /// Which tool was called.
    pub tool_name: String,
    /// The tool result.
    pub result: ToolResult,
    /// Whether knowledge was injected into state.
    pub injected: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Mamba3Agent
// ═══════════════════════════════════════════════════════════════════════════

/// The full SSD agent: Mamba-3 JEPA perception + tool-use delegation.
///
/// This is the "real model" — a persistent agent with:
/// - **Fast thinking**: Mamba-3 SSM processes inputs in O(N) with persistent state
/// - **Slow thinking**: Tool delegation when the confusion probe fires
/// - **Continuous identity**: SSM state never resets, accumulating personality
/// - **Selective output**: Only decodes text when state drift warrants it
pub struct Mamba3Agent {
    /// The underlying Mamba3-JEPA world model.
    pub model: Mamba3Jepa,
    /// Agent behavior configuration.
    pub agent_config: AgentConfig,
    /// Registered tools for delegation.
    pub tools: ToolRegistry,
    /// State injector for tool results (separate from recursion layer's injector).
    tool_injector: StateInjector,
    /// Selective decoder for streaming output.
    selective_decoder: SelectiveDecoder,
    /// Disk-backed dual-stream memory — standard, always attempted.
    /// Memory is the foundation of the World Model, not an optional plugin.
    /// None only if filesystem init failed (degraded mode with loud warning).
    memory: Option<VisualMemory>,
    /// Current step counter.
    step_count: usize,
    /// Last embedding produced (stored as f16 to halve memory).
    last_embedding: Option<Vec<f16>>,
    /// VisualGate: caches ViT output, skips re-encoding on static frames.
    visual_gate: VisualGate,
    /// Last tool knowledge for virtual token feedback (Memento fix).
    /// Fed back through Mamba on the next perceive() call so tool results
    /// get written into the SSM hidden state, not just the output embedding.
    last_tool_knowledge: Option<Vec<f32>>,
    /// Knowledge dimension for feedback (matches recursion config).
    feedback_dim: usize,
}

impl Mamba3Agent {
    /// Build a new agent from model and agent configs.
    pub fn new(model_config: Mamba3JepaConfig, agent_config: AgentConfig) -> Self {
        let selective_decoder = SelectiveDecoder::new(agent_config.selective_decode.clone());
        let knowledge_dim = model_config
            .recursion
            .knowledge_dim
            .max(model_config.shared_embed_dim);
        let tool_injector = StateInjector::new(model_config.shared_embed_dim, knowledge_dim);
        let model = Mamba3Jepa::new(model_config);

        // Memory is standard — always attempted, never opt-in.
        // If the filesystem fails, the agent degrades gracefully (no crash)
        // but logs a loud warning because this is NOT expected.
        let memory = match VisualMemory::open_index_only(agent_config.memory.clone()) {
            Ok(mem) => Some(mem),
            Err(e) => {
                eprintln!("[Xura] WARNING: failed to open VisualMemory: {}", e);
                eprintln!("[Xura] WARNING: agent running in DEGRADED MODE — no memory persistence!");
                eprintln!("[Xura] WARNING: the agent will forget everything between sessions.");
                None
            }
        };

        Self {
            model,
            agent_config,
            tools: ToolRegistry::new(),
            tool_injector,
            selective_decoder,
            memory,
            step_count: 0,
            last_embedding: None,
            visual_gate: VisualGate::new(),
            last_tool_knowledge: None,
            feedback_dim: knowledge_dim,
        }
    }

    /// Register a tool with the agent.
    pub fn register_tool(&mut self, tool: Box<dyn Tool>) {
        self.tools.register(tool);
    }

    /// Run a single agent step: perceive → monitor → delegate → inject → act.
    ///
    /// This is the core SSD loop. Each call:
    /// 1. Encodes the input through Mamba-JEPA to get an embedding
    /// 2. Checks the confusion monitor on the predictor's hidden state
    /// 3. If confused and tools are available, delegates to the best tool
    /// 4. Injects tool knowledge back into the embedding
    /// 5. Optionally decodes the embedding to text tokens
    /// 6. Stores the interaction in episodic memory
    pub fn step(&mut self, input: &AgentInput) -> AgentOutput {
        self.step_count += 1;
        let step_num = self.step_count;

        // ── 1. PERCEIVE ──────────────────────────────────────────────
        // Encode input through the JEPA model to get an embedding.
        let (embedding, did_perceive) = self.perceive(input);

        // ── 2. MONITOR ───────────────────────────────────────────────
        // Check confusion score from the recursion layer.
        let confusion_score = self.get_confusion_score();

        // Check state drift via selective decoder.
        let state_drift = if did_perceive {
            self.selective_decoder.should_decode(&embedding);
            self.selective_decoder.current_drift()
        } else {
            0.0
        };

        // ── 3. DELEGATE ──────────────────────────────────────────────
        // If confused and tools available, call tools.
        let mut tool_calls = Vec::new();
        let mut injected_embedding = embedding.clone();

        if confusion_score > self.get_confusion_threshold() && !self.tools.is_empty() {
            let max_calls = self.agent_config.max_tool_calls_per_step;
            for call_idx in 0..max_calls {
                let request = self.build_tool_request(&injected_embedding, input, call_idx as u8);

                if let Some(tool) = self.tools.select_tool(&request.query_embedding) {
                    let result = tool.execute(&request);

                    // ── 4. INJECT ────────────────────────────────────
                    let injected = if result.success && !result.knowledge_embedding.is_empty() {
                        // Store raw knowledge for virtual token feedback on next step.
                        // This fixes the "Memento" problem: tool results get written
                        // into Mamba's hidden state via perceive() on the next call.
                        self.last_tool_knowledge = Some(
                            Self::pad_or_truncate(&result.knowledge_embedding, self.feedback_dim),
                        );
                        injected_embedding =
                            self.inject_knowledge(&injected_embedding, &result.knowledge_embedding);
                        true
                    } else {
                        false
                    };

                    let record = ToolCallRecord {
                        tool_name: result.tool_name.clone(),
                        result,
                        injected,
                    };
                    tool_calls.push(record);

                    // Re-check confusion — maybe one call was enough
                    // (simplified: just break after first successful injection)
                    if injected {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // ── 5. ACT ───────────────────────────────────────────────────
        // Decide whether to decode and produce output tokens.
        let (response_tokens, did_decode) = self.act(&injected_embedding, state_drift);

        // ── 6. REMEMBER ──────────────────────────────────────────────
        // Store in disk-backed memory (if enabled).
        self.remember(input, &injected_embedding, &response_tokens, step_num);
        self.last_embedding = Some(injected_embedding.iter().map(|&v| f16::from_f32(v)).collect());

        AgentOutput {
            embedding: injected_embedding,
            response_tokens,
            tool_calls,
            did_decode,
            confusion_score,
            state_drift,
            step_number: step_num,
        }
    }

    /// Reset the agent's state (clear Mamba state, flush memory, etc.).
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.last_embedding = None;
        // Flush disk memory before resetting
        if let Some(ref mut mem) = self.memory {
            if let Err(e) = mem.flush() {
                eprintln!("[Xura] WARNING: memory flush on reset failed: {}", e);
            }
        }
        self.selective_decoder.reset();
        self.visual_gate.reset();
        self.last_tool_knowledge = None;
        if let Some(ref mut rec) = self.model.predictor.recursion {
            rec.reset();
        }
    }

    /// Get the current step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the number of entries written to disk-backed memory.
    /// Returns 0 if running in degraded mode (memory init failed).
    pub fn memory_entry_count(&self) -> u64 {
        self.memory.as_ref().map_or(0, |m| m.index_count())
    }

    // ─── Internal methods ────────────────────────────────────────────

    /// Perceive: encode input through Mamba-JEPA.
    ///
    /// Uses VisualGate to skip ViT when the image hasn't changed ("Blink").
    /// Passes last_tool_knowledge as virtual token feedback so Mamba reads
    /// tool results into its hidden state ("Memento" fix).
    fn perceive(&mut self, input: &AgentInput) -> (Vec<f32>, bool) {
        let embed_dim = self.model.config.shared_embed_dim;

        // Take feedback from last step's tool use (consumed once)
        let feedback = self.last_tool_knowledge.take();
        let fb_ref = feedback.as_deref();

        if let Some(ref image) = input.image {
            let num_patches = self.model.config.vit.num_patches();

            // VisualGate: check if image changed since last step
            let visual_tokens = if let Some((cached, _np)) = self.visual_gate.check(image) {
                cached
            } else {
                // Cache miss: run ViT and store result
                let tokens = self.model.encode_vision(image, input.image_h, input.image_w);
                self.visual_gate.store(image, &tokens, num_patches);
                tokens
            };

            // Run predictor with cached visual tokens + optional feedback
            let output = self.model.infer_with_feedback(
                Some(&visual_tokens),
                num_patches,
                &input.query_tokens,
                fb_ref,
                InferenceMode::Embedding,
            );
            match output {
                InferenceOutput::Embedding(emb) => (emb, true),
                _ => (vec![0.0f32; embed_dim], false),
            }
        } else {
            // Text-only fast-path with optional feedback
            let output = self.model.infer_with_feedback(
                None,
                0,
                &input.query_tokens,
                fb_ref,
                InferenceMode::Embedding,
            );
            match output {
                InferenceOutput::Embedding(emb) => (emb, true),
                _ => (vec![0.0f32; embed_dim], false),
            }
        }
    }

    /// Get the confusion score from the recursion layer.
    fn get_confusion_score(&self) -> f32 {
        if let Some(ref rec) = self.model.predictor.recursion {
            rec.monitor.current_score()
        } else {
            0.0
        }
    }

    /// Get the confusion threshold.
    fn get_confusion_threshold(&self) -> f32 {
        self.model.config.recursion.confusion_threshold
    }

    /// Build a tool request from the current state.
    fn build_tool_request(&self, embedding: &[f32], input: &AgentInput, depth: u8) -> ToolRequest {
        let query_text = format!("step:{} tokens:{:?}", self.step_count, &input.query_tokens);

        ToolRequest {
            tool_name: String::new(), // will be filled by tool selection
            query_embedding: embedding.to_vec(),
            query_text,
            depth,
            params: std::collections::HashMap::new(),
        }
    }

    /// Inject knowledge from a tool result into the embedding.
    fn inject_knowledge(&self, embedding: &[f32], knowledge: &[f32]) -> Vec<f32> {
        // Use the tool injector: h_new = h + W_inject · V_knowledge
        // The injector operates on (batch*seq_len, d_model) shaped data.
        // For a single embedding, batch=1, seq_len=1.
        self.tool_injector.inject(embedding, knowledge, 1, 1)
    }

    /// Decide whether to decode and produce response tokens.
    fn act(&mut self, embedding: &[f32], state_drift: f32) -> (Vec<usize>, bool) {
        // Selective decoding: only decode if state drift is significant
        let should_decode = if self.agent_config.selective_decoding {
            state_drift > self.agent_config.selective_decode.drift_threshold
        } else {
            true // always decode if selective decoding is off
        };

        if !should_decode {
            return (Vec::new(), false);
        }

        // Decode embedding → text tokens
        let tokens = self.model.y_decoder.generate(
            embedding,
            self.agent_config.max_response_tokens,
            self.agent_config.bos_token,
        );

        // Truncate at EOS if present
        let eos = self.agent_config.eos_token;
        let tokens = if let Some(pos) = tokens.iter().position(|&t| t == eos) {
            tokens[..=pos].to_vec()
        } else {
            tokens
        };

        (tokens, true)
    }

    /// Pad or truncate a vector to a target length.
    fn pad_or_truncate(v: &[f32], target_len: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; target_len];
        let copy_len = v.len().min(target_len);
        out[..copy_len].copy_from_slice(&v[..copy_len]);
        out
    }

    /// Store interaction in disk-backed memory.
    ///
    /// Writes the embedding, response tokens, and (optionally) a video frame
    /// to the dual-stream memory system.
    fn remember(
        &mut self,
        input: &AgentInput,
        embedding: &[f32],
        response_tokens: &[usize],
        step: usize,
    ) {
        let mem = match self.memory.as_mut() {
            Some(m) => m,
            None => return, // degraded mode — no memory
        };

        // Convert f32 image to RGB bytes for ffmpeg (if video stream is active)
        let rgb_bytes = if mem.has_video() {
            input.image.as_ref().map(|img| {
                let c = self.model.config.vit.in_channels;
                f32_image_to_rgb_bytes(img, c, input.image_h, input.image_w)
            })
        } else {
            None
        };

        if let Err(e) = mem.append(
            rgb_bytes.as_deref(),
            embedding,
            response_tokens,
            step,
        ) {
            eprintln!("[Xura] WARNING: memory append failed at step {}: {}", step, e);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, Mamba3JepaConfig, RecursionConfig, VisualMemoryConfig};
    use crate::tools::{EchoTool, MemorySearchTool};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Monotonic counter to give each test a unique memory directory.
    static TEST_ID: AtomicUsize = AtomicUsize::new(0);

    fn test_agent_config() -> AgentConfig {
        let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
        let mut cfg = AgentConfig::tiny();
        cfg.memory = VisualMemoryConfig::test(&format!("agent_{}", id));
        cfg
    }

    fn make_agent() -> Mamba3Agent {
        let model_config = Mamba3JepaConfig::tiny();
        Mamba3Agent::new(model_config, test_agent_config())
    }

    fn make_agent_with_recursion() -> Mamba3Agent {
        let mut model_config = Mamba3JepaConfig::tiny();
        model_config.recursion = RecursionConfig {
            enabled: true,
            knowledge_dim: 32,
            confusion_threshold: 0.0, // always trigger for testing
            confusion_dims: 16,
            inject_after_layer: 0,
            max_depth: 1,
            smoothing: 1.0,
        };
        Mamba3Agent::new(model_config, test_agent_config())
    }

    #[test]
    fn test_agent_creation() {
        let agent = make_agent();
        assert_eq!(agent.step_count(), 0);
        assert_eq!(agent.memory_entry_count(), 0);
    }

    #[test]
    fn test_agent_text_step() {
        let mut agent = make_agent();
        let input = AgentInput::text(vec![1, 2, 3]);

        let output = agent.step(&input);
        assert_eq!(output.step_number, 1);
        assert!(!output.embedding.is_empty());
        assert!(output.did_decode);
        assert!(output.tool_calls.is_empty());
        assert_eq!(agent.step_count(), 1);
        assert_eq!(agent.memory_entry_count(), 1);
    }

    #[test]
    fn test_agent_vision_text_step() {
        let mut agent = make_agent();
        let config = &agent.model.config;
        let h = config.vit.image_size;
        let w = config.vit.image_size;
        let image = vec![0.5f32; config.vit.in_channels * h * w];

        let input = AgentInput::vision_text(image, h, w, vec![1, 2]);
        let output = agent.step(&input);

        assert_eq!(output.step_number, 1);
        assert!(!output.embedding.is_empty());
        assert!(output.did_decode);
    }

    #[test]
    fn test_agent_multiple_steps() {
        let mut agent = make_agent();

        for i in 0..5 {
            let input = AgentInput::text(vec![i % 10, (i + 1) % 10]);
            let output = agent.step(&input);
            assert_eq!(output.step_number, i + 1);
        }

        assert_eq!(agent.step_count(), 5);
        assert_eq!(agent.memory_entry_count(), 5);
    }

    #[test]
    fn test_agent_persistent_state() {
        let mut agent = make_agent();

        // Step 1
        let input1 = AgentInput::text(vec![1, 2, 3]);
        let out1 = agent.step(&input1);

        // Step 2 — state should be different because Mamba state accumulated
        let input2 = AgentInput::text(vec![4, 5, 6]);
        let out2 = agent.step(&input2);

        // Embeddings should differ (different inputs + accumulated state)
        assert_ne!(out1.embedding, out2.embedding);
    }

    #[test]
    fn test_agent_with_echo_tool() {
        let mut agent = make_agent_with_recursion();
        agent.register_tool(Box::new(EchoTool::new(32)));

        let input = AgentInput::text(vec![1, 2, 3]);
        let output = agent.step(&input);

        // With recursion enabled (threshold=0), confusion always triggers
        // so the echo tool should be called
        assert!(!output.tool_calls.is_empty());
        assert_eq!(output.tool_calls[0].tool_name, "echo");
        assert!(output.tool_calls[0].result.success);
        assert!(output.tool_calls[0].injected);
    }

    #[test]
    fn test_agent_with_memory_tool() {
        let mut agent = make_agent_with_recursion();

        let dim = agent.model.config.shared_embed_dim;
        let mut memory = MemorySearchTool::new(dim);
        memory.add_entry(vec![1.0; dim], vec![0.5; dim], "knowledge_1".into());
        agent.register_tool(Box::new(memory));

        let input = AgentInput::text(vec![1, 2, 3]);
        let output = agent.step(&input);

        assert!(!output.tool_calls.is_empty());
        assert_eq!(output.tool_calls[0].tool_name, "memory_search");
        assert!(output.tool_calls[0].result.success);
        assert!(output.tool_calls[0].result.summary.contains("knowledge_1"));
    }

    #[test]
    fn test_agent_no_tools_no_delegation() {
        let mut agent = make_agent_with_recursion();
        // No tools registered

        let input = AgentInput::text(vec![1, 2, 3]);
        let output = agent.step(&input);

        // No tools → no delegation even if confused
        assert!(output.tool_calls.is_empty());
    }

    #[test]
    fn test_agent_selective_decoding() {
        let model_config = Mamba3JepaConfig::tiny();
        let mut agent_config = test_agent_config();
        agent_config.selective_decoding = true;
        agent_config.selective_decode.drift_threshold = 999.0; // never decode

        let mut agent = Mamba3Agent::new(model_config, agent_config);

        let input = AgentInput::text(vec![1, 2, 3]);
        let output = agent.step(&input);

        // Drift threshold is absurdly high → should NOT decode
        assert!(!output.did_decode);
        assert!(output.response_tokens.is_empty());
    }

    #[test]
    fn test_agent_reset() {
        let mut agent = make_agent();

        agent.step(&AgentInput::text(vec![1, 2]));
        agent.step(&AgentInput::text(vec![3, 4]));
        assert_eq!(agent.step_count(), 2);

        agent.reset();
        assert_eq!(agent.step_count(), 0);
    }

    // ─── Regression tests for architectural fixes ─────────────────

    /// Memento fix: tool knowledge stored for virtual token feedback.
    /// After a tool call, last_tool_knowledge should be set, and on
    /// the next step it should be consumed (fed through Mamba).
    #[test]
    fn test_memento_tool_feedback_stored() {
        let mut agent = make_agent_with_recursion();
        agent.register_tool(Box::new(EchoTool::new(32)));

        // Step 1: tool fires, knowledge should be stored for next step
        let out1 = agent.step(&AgentInput::text(vec![1, 2, 3]));
        assert!(!out1.tool_calls.is_empty());
        assert!(out1.tool_calls[0].injected);
        // After step(), last_tool_knowledge is set (will be consumed next perceive)
        assert!(agent.last_tool_knowledge.is_some());

        // Step 2: feedback is consumed during perceive(), influencing the output.
        // Same query tokens as step 1, but the virtual feedback token should
        // cause a different embedding — proving the tool result persisted.
        let out2 = agent.step(&AgentInput::text(vec![1, 2, 3]));
        // Both outputs should be finite (feedback didn't cause NaN)
        assert!(out1.embedding.iter().all(|v| v.is_finite()));
        assert!(out2.embedding.iter().all(|v| v.is_finite()));
        // Embedding must differ — the feedback virtual token changed the result
        assert_ne!(out1.embedding, out2.embedding, "feedback should influence step 2 embedding");
    }

    /// Memento fix: feedback_proj exists when recursion is enabled.
    #[test]
    fn test_memento_feedback_proj_exists() {
        let agent = make_agent_with_recursion();
        assert!(agent.model.predictor.has_feedback_proj());
    }

    /// Memento fix: feedback_proj is None when recursion is disabled.
    #[test]
    fn test_memento_no_feedback_proj_without_recursion() {
        let agent = make_agent();
        assert!(!agent.model.predictor.has_feedback_proj());
    }

    /// Retina Burn fix: VisualGate caches ViT output on static frames.
    #[test]
    fn test_visual_gate_cache_hit() {
        let mut agent = make_agent();
        let h = agent.model.config.vit.image_size;
        let w = agent.model.config.vit.image_size;
        let channels = agent.model.config.vit.in_channels;
        let image = vec![0.5f32; channels * h * w];

        // Step 1: cache miss (first time seeing this image)
        let input = AgentInput::vision_text(image.clone(), h, w, vec![1, 2]);
        agent.step(&input);
        assert_eq!(agent.visual_gate.misses, 1);
        assert_eq!(agent.visual_gate.hits, 0);

        // Step 2: same image → cache hit (ViT skipped)
        let input2 = AgentInput::vision_text(image.clone(), h, w, vec![3, 4]);
        agent.step(&input2);
        assert_eq!(agent.visual_gate.hits, 1);

        // Step 3: different image → cache miss
        let image2 = vec![0.9f32; channels * h * w];
        let input3 = AgentInput::vision_text(image2, h, w, vec![5, 6]);
        agent.step(&input3);
        assert_eq!(agent.visual_gate.misses, 2);
    }

    /// Retina Burn fix: text-only steps don't touch the visual gate.
    #[test]
    fn test_visual_gate_text_only_no_cache() {
        let mut agent = make_agent();
        agent.step(&AgentInput::text(vec![1, 2]));
        assert_eq!(agent.visual_gate.hits, 0);
        assert_eq!(agent.visual_gate.misses, 0);
    }

    /// Precision Bloat fix: last_embedding stored as f16.
    #[test]
    fn test_f16_last_embedding() {
        let mut agent = make_agent();
        agent.step(&AgentInput::text(vec![1, 2, 3]));
        assert!(agent.last_embedding.is_some());

        let stored = agent.last_embedding.as_ref().unwrap();
        let restored: Vec<f32> = stored.iter().map(|v| v.to_f32()).collect();
        assert!(!restored.is_empty());
        assert!(restored.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_agent_output_finite() {
        let mut agent = make_agent_with_recursion();
        agent.register_tool(Box::new(EchoTool::new(32)));

        for _ in 0..3 {
            let input = AgentInput::text(vec![1, 2, 3]);
            let output = agent.step(&input);

            assert!(output.embedding.iter().all(|v| v.is_finite()));
            assert!(output.confusion_score.is_finite());
            assert!(output.state_drift.is_finite());
        }
    }
}
