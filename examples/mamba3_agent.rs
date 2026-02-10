//! Mamba3 Agent — Full SSD (State-Space Delegation) Loop Demo
//!
//! Demonstrates the complete agent cycle:
//!   1. Build agent with model + tools
//!   2. Register knowledge tools (memory search, echo)
//!   3. Run multi-turn conversation with persistent Mamba state
//!   4. Observe confusion monitoring, tool delegation, state injection
//!   5. Show selective decoding (only respond when state drifts)
//!
//! Run with:
//!   cargo run --example mamba3_agent --release
//!
//! Architecture:
//!
//!   ┌──────────────────────────────────────────────────────────┐
//!   │                    Mamba3Agent                            │
//!   │                                                          │
//!   │  Input ──► ViT ──► Mamba-3 Predictor ──► Embedding      │
//!   │                        │                     │           │
//!   │                  ConfusionMonitor        StateInjector    │
//!   │                        │                     ▲           │
//!   │                  confused?──yes──► Tool ──────┘           │
//!   │                        │                                 │
//!   │                        no                                │
//!   │                        │                                 │
//!   │                  SelectiveDecoder                        │
//!   │                        │                                 │
//!   │                  drift > θ ?──yes──► Y-Decoder ──► Text  │
//!   │                        │                                 │
//!   │                        no ──► (silence)                  │
//!   │                                                          │
//!   │  Episodic Memory: stores (embedding, response) pairs     │
//!   │  SSM State: never resets → continuous stream of thought  │
//!   └──────────────────────────────────────────────────────────┘

use std::time::Instant;

use xura_vljepa::{
    Mamba3Agent, AgentInput, AgentConfig,
    Mamba3JepaConfig, RecursionConfig,
    MemorySearchTool, EchoTool,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Mamba3 Agent — State-Space Delegation (SSD)          ║");
    println!("║  Persistent Mamba-3 state + tool-based slow thinking       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. Configuration ─────────────────────────────────────────────
    //
    // Model: tiny preset (for demo speed)
    // Recursion: enabled, low threshold so delegation fires often
    // Agent: decode always, persistent state

    let mut model_config = Mamba3JepaConfig::tiny();
    model_config.recursion = RecursionConfig {
        enabled: true,
        knowledge_dim: model_config.shared_embed_dim,
        confusion_threshold: 0.3,
        confusion_dims: model_config.predictor.d_model.min(16),
        inject_after_layer: 0,
        max_depth: 1,
        smoothing: 0.8,
    };

    let agent_config = AgentConfig {
        max_tool_calls_per_step: 3,
        max_response_tokens: 16,
        bos_token: 0,
        eos_token: 1,
        selective_decoding: false, // decode every step for demo
        ..AgentConfig::default()
    };

    let dim = model_config.shared_embed_dim;

    println!("── Configuration ──");
    println!("  Model: tiny (d_model={}, layers={}, embed_dim={})",
        model_config.predictor.d_model,
        model_config.predictor.n_layers,
        dim);
    println!("  Recursion: threshold={}, inject_after_layer={}",
        model_config.recursion.confusion_threshold,
        model_config.recursion.inject_after_layer);
    println!("  Agent: max_tool_calls={}, max_tokens={}, persistent_state={}",
        agent_config.max_tool_calls_per_step,
        agent_config.max_response_tokens,
        agent_config.persistent_state);
    println!();

    // ── 2. Build Agent ───────────────────────────────────────────────

    let t0 = Instant::now();
    let mut agent = Mamba3Agent::new(model_config.clone(), agent_config);
    println!("Agent built in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // ── 3. Register Tools ────────────────────────────────────────────
    //
    // In production, tools could be:
    //   - Vector DB (RAG-style retrieval)
    //   - Code interpreter
    //   - Web search API
    //   - Sub-agent (another Mamba3Agent!)
    //
    // Here we use built-in MemorySearchTool + EchoTool.

    // Populate knowledge base
    let mut memory = MemorySearchTool::new(dim);

    // Synthetic knowledge entries (in production: embed real documents)
    let knowledge = vec![
        ("rust programming", 0),
        ("neural networks", 1),
        ("quantum computing", 2),
        ("game development", 3),
        ("cryptography", 4),
    ];

    for (label, seed) in &knowledge {
        let mut key = vec![0.0f32; dim];
        let mut val = vec![0.0f32; dim];
        // Create distinct embeddings for each topic
        for d in 0..dim {
            let t = (seed * dim + d) as f32 * 0.1;
            key[d] = (t * 1.7).sin();
            val[d] = (t * 2.3 + 1.0).cos() * 0.5;
        }
        memory.add_entry(key, val, label.to_string());
    }

    agent.register_tool(Box::new(memory));
    agent.register_tool(Box::new(EchoTool::new(dim)));

    println!("Tools registered: {:?}", agent.tools.names());
    println!("Knowledge base: {} entries", 5);
    println!();

    // ── 4. Multi-Turn Conversation ───────────────────────────────────
    //
    // Each turn feeds token IDs (simulating tokenized text) to the agent.
    // The Mamba-3 SSM state persists across turns, so each response is
    // informed by the full conversation history.

    println!("═══ Multi-Turn Agent Conversation ═══");
    println!();

    // Simulated conversation turns (token IDs)
    let turns: Vec<(&str, Vec<usize>)> = vec![
        ("Hello, who are you?",                    vec![5, 12, 8, 3]),
        ("Tell me about neural networks",          vec![7, 15, 2, 9, 11]),
        ("How does backpropagation work?",         vec![4, 18, 6, 14, 10, 3]),
        ("Now explain quantum computing",          vec![9, 20, 13, 7, 5]),
        ("Compare the two topics",                 vec![3, 16, 8, 12, 19, 6]),
        ("Write a summary",                        vec![11, 4, 17]),
    ];

    let mut total_time = std::time::Duration::ZERO;
    let mut total_tool_calls = 0;
    let mut total_delegations = 0;

    for (i, (description, tokens)) in turns.iter().enumerate() {
        println!("┌─ Turn {} ─────────────────────────────────────────────┐", i + 1);
        println!("│  User: \"{}\"", description);
        println!("│  Tokens: {:?}", tokens);

        let input = AgentInput::text(tokens.clone());
        let t_step = Instant::now();
        let output = agent.step(&input);
        let step_time = t_step.elapsed();
        total_time += step_time;

        println!("│");
        println!("│  Confusion score:  {:.4}", output.confusion_score);
        println!("│  State drift:      {:.4}", output.state_drift);
        println!("│  Tool calls:       {}", output.tool_calls.len());

        for tc in &output.tool_calls {
            println!("│    → {} (injected: {}) \"{}\"",
                tc.tool_name, tc.injected, tc.result.summary);
            total_tool_calls += 1;
            if tc.injected { total_delegations += 1; }
        }

        println!("│  Did decode:       {}", output.did_decode);
        println!("│  Response tokens:  {} tokens", output.response_tokens.len());
        if !output.response_tokens.is_empty() {
            println!("│  Token IDs:        {:?}",
                &output.response_tokens[..output.response_tokens.len().min(10)]);
        }
        println!("│  Embedding norm:   {:.4}", vec_norm(&output.embedding));
        println!("│  Step time:        {:.2}ms", step_time.as_secs_f64() * 1000.0);
        println!("└──────────────────────────────────────────────────────┘");
        println!();
    }

    // ── 5. Vision + Text Turn ────────────────────────────────────────

    println!("═══ Vision + Text Turn ═══");
    println!();

    let h = model_config.vit.image_size;
    let w = model_config.vit.image_size;
    let image = vec![0.5f32; model_config.vit.in_channels * h * w];

    let input = AgentInput::vision_text(image, h, w, vec![1, 2, 3]);
    let t_vis = Instant::now();
    let output = agent.step(&input);
    let vis_time = t_vis.elapsed();
    total_time += vis_time;

    println!("  Image: {}×{}, query: [1, 2, 3]", h, w);
    println!("  Confusion: {:.4}, Drift: {:.4}", output.confusion_score, output.state_drift);
    println!("  Tool calls: {}, Decoded: {}", output.tool_calls.len(), output.did_decode);
    println!("  Embedding norm: {:.4}", vec_norm(&output.embedding));
    println!("  Time: {:.2}ms", vis_time.as_secs_f64() * 1000.0);
    println!();

    // ── 6. Selective Decoding Demo ───────────────────────────────────

    println!("═══ Selective Decoding Demo ═══");
    println!();
    println!("  (Rebuilding agent with selective_decoding=true, threshold=0.5)");

    let sel_agent_config = AgentConfig {
        selective_decoding: true,
        max_response_tokens: 16,
        bos_token: 0,
        eos_token: 1,
        ..AgentConfig::default()
    };

    let mut sel_agent = Mamba3Agent::new(model_config.clone(), sel_agent_config);
    sel_agent.register_tool(Box::new(EchoTool::new(dim)));

    // Send several similar inputs — agent should stay silent
    // Then send a very different input — agent should speak
    let sel_turns = vec![
        ("same topic",    vec![1, 2, 3]),
        ("same topic",    vec![1, 2, 3]),
        ("same topic",    vec![1, 2, 3]),
        ("TOPIC CHANGE!", vec![99, 88, 77, 66, 55]),
        ("TOPIC CHANGE!", vec![99, 88, 77, 66, 55]),
    ];

    for (desc, tokens) in &sel_turns {
        let input = AgentInput::text(tokens.clone());
        let output = sel_agent.step(&input);
        println!("  \"{}\" → drift={:.4} decoded={} tokens={}",
            desc, output.state_drift, output.did_decode, output.response_tokens.len());
    }
    println!();

    // ── 7. Agent State Inspection ────────────────────────────────────

    println!("═══ Agent State ═══");
    println!();
    println!("  Total steps (main agent):       {}", agent.step_count());
    println!("  Episodic memory entries:         {}", agent.episodic_memory_len());
    println!("  Total tool calls:                {}", total_tool_calls);
    println!("  Successful delegations:          {}", total_delegations);
    println!("  Total inference time:            {:.1}ms", total_time.as_secs_f64() * 1000.0);
    println!("  Avg time per step:               {:.1}ms",
        total_time.as_secs_f64() * 1000.0 / (turns.len() + 1) as f64);
    println!();

    // ── 8. Reset & Summary ───────────────────────────────────────────

    agent.reset();
    assert_eq!(agent.step_count(), 0);
    println!("  Agent reset: step_count={}, memory={}", agent.step_count(), agent.episodic_memory_len());
    println!();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Agent Summary                                             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                            ║");
    println!("║  The Mamba3 Agent implements the full SSD loop:            ║");
    println!("║                                                            ║");
    println!("║  PERCEIVE: Mamba-3 JEPA encodes image+text → embedding    ║");
    println!("║  MONITOR:  Confusion probe on predictor hidden state      ║");
    println!("║  DELEGATE: Tool call when confused (memory, echo, etc.)   ║");
    println!("║  INJECT:   h_new = h + W_inject · V_knowledge             ║");
    println!("║  ACT:      Decode to text (selective: only on drift)      ║");
    println!("║  REMEMBER: Store in episodic memory for future recall     ║");
    println!("║                                                            ║");
    println!("║  Key properties:                                           ║");
    println!("║  • O(N) perception via Mamba-3 SSM (no attention)         ║");
    println!("║  • Persistent state = continuous identity                  ║");
    println!("║  • Pluggable tools via Tool trait                          ║");
    println!("║  • Selective output = speak only when warranted            ║");
    println!("║  • Episodic memory for long-horizon context                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    println!("Next steps:");
    println!("  1. Load real ViT weights: load_vit_weights(\"vjepa2-vit-l.safetensors\")");
    println!("  2. Implement a real tokenizer (SentencePiece / BPE)");
    println!("  3. Plug in vector DB tool (Qdrant, Pinecone, etc.) via Tool trait");
    println!("  4. Add code interpreter tool for agentic coding");
    println!("  5. Train with kore-autograd on real data");
    println!("  6. Scale to Mamba3JepaConfig::small() for production");
    println!("  7. Add sub-agent tool (another Mamba3Agent as a tool!)");
}

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
