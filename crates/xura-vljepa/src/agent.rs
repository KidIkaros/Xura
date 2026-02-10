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

use crate::config::{AgentConfig, Mamba3JepaConfig};
use crate::recursion::StateInjector;
use crate::selective_decode::SelectiveDecoder;
use crate::tools::{Tool, ToolRegistry, ToolRequest, ToolResult};
use crate::vljepa::{InferenceMode, InferenceOutput, Mamba3Jepa};

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
    /// Episodic memory: recent (embedding, response) pairs.
    episodic_memory: Vec<EpisodicEntry>,
    /// Maximum episodic memory entries.
    max_episodic_entries: usize,
    /// Current step counter.
    step_count: usize,
    /// Last embedding produced (for state drift tracking).
    last_embedding: Option<Vec<f32>>,
}

/// An entry in the agent's episodic memory.
#[derive(Clone, Debug)]
struct EpisodicEntry {
    embedding: Vec<f32>,
    response_tokens: Vec<usize>,
    step: usize,
}

impl Mamba3Agent {
    /// Build a new agent from model and agent configs.
    pub fn new(
        model_config: Mamba3JepaConfig,
        agent_config: AgentConfig,
    ) -> Self {
        let selective_decoder = SelectiveDecoder::new(agent_config.selective_decode.clone());
        let knowledge_dim = model_config.recursion.knowledge_dim.max(
            model_config.shared_embed_dim,
        );
        let tool_injector = StateInjector::new(
            model_config.shared_embed_dim,
            knowledge_dim,
        );
        let model = Mamba3Jepa::new(model_config);

        Self {
            model,
            agent_config,
            tools: ToolRegistry::new(),
            tool_injector,
            selective_decoder,
            episodic_memory: Vec::new(),
            max_episodic_entries: 1024,
            step_count: 0,
            last_embedding: None,
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
                        injected_embedding = self.inject_knowledge(
                            &injected_embedding,
                            &result.knowledge_embedding,
                        );
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
        // Store in episodic memory.
        self.remember(&injected_embedding, &response_tokens, step_num);
        self.last_embedding = Some(injected_embedding.clone());

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

    /// Reset the agent's state (clear Mamba state, episodic memory, etc.).
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.last_embedding = None;
        self.episodic_memory.clear();
        self.selective_decoder.reset();
        if let Some(ref mut rec) = self.model.predictor.recursion {
            rec.reset();
        }
    }

    /// Get the current step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the episodic memory entries.
    pub fn episodic_memory_len(&self) -> usize {
        self.episodic_memory.len()
    }

    // ─── Internal methods ────────────────────────────────────────────

    /// Perceive: encode input through Mamba-JEPA.
    ///
    /// Uses the text-only fast-path when no image is provided,
    /// skipping the ViT X-Encoder entirely to avoid unnecessary compute.
    fn perceive(&mut self, input: &AgentInput) -> (Vec<f32>, bool) {
        let embed_dim = self.model.config.shared_embed_dim;

        if let Some(ref image) = input.image {
            // Vision + text: full VLJEPA path (ViT → Predictor)
            let output = self.model.infer(
                image,
                &input.query_tokens,
                input.image_h,
                input.image_w,
                InferenceMode::Embedding,
            );
            match output {
                InferenceOutput::Embedding(emb) => (emb, true),
                _ => (vec![0.0f32; embed_dim], false),
            }
        } else {
            // Text-only fast-path: skip ViT, feed query tokens directly to Predictor
            let output = self.model.infer_text_only(
                &input.query_tokens,
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
    fn build_tool_request(
        &self,
        embedding: &[f32],
        input: &AgentInput,
        depth: u8,
    ) -> ToolRequest {
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
    fn inject_knowledge(
        &self,
        embedding: &[f32],
        knowledge: &[f32],
    ) -> Vec<f32> {
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

    /// Store interaction in episodic memory.
    fn remember(&mut self, embedding: &[f32], response_tokens: &[usize], step: usize) {
        let entry = EpisodicEntry {
            embedding: embedding.to_vec(),
            response_tokens: response_tokens.to_vec(),
            step,
        };
        self.episodic_memory.push(entry);

        // Evict oldest if over capacity
        if self.episodic_memory.len() > self.max_episodic_entries {
            self.episodic_memory.remove(0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, Mamba3JepaConfig, RecursionConfig};
    use crate::tools::{EchoTool, MemorySearchTool};

    fn make_agent() -> Mamba3Agent {
        let model_config = Mamba3JepaConfig::tiny();
        let agent_config = AgentConfig::tiny();
        Mamba3Agent::new(model_config, agent_config)
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
        let agent_config = AgentConfig::tiny();
        Mamba3Agent::new(model_config, agent_config)
    }

    #[test]
    fn test_agent_creation() {
        let agent = make_agent();
        assert_eq!(agent.step_count(), 0);
        assert_eq!(agent.episodic_memory_len(), 0);
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
        assert_eq!(agent.episodic_memory_len(), 1);
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
        assert_eq!(agent.episodic_memory_len(), 5);
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
        memory.add_entry(
            vec![1.0; dim],
            vec![0.5; dim],
            "knowledge_1".into(),
        );
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
        let mut agent_config = AgentConfig::tiny();
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
        assert_eq!(agent.episodic_memory_len(), 2);

        agent.reset();
        assert_eq!(agent.step_count(), 0);
        assert_eq!(agent.episodic_memory_len(), 0);
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
