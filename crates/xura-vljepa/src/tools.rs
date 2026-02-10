//! Agent Tool System — pluggable tools for the SSD agent loop.
//!
//! Tools are the "slow-thinking" capabilities that the agent delegates to
//! when its internal Mamba state isn't enough. Each tool receives a structured
//! request and returns a structured result.
//!
//! Built-in tools:
//! - **MemorySearchTool**: vector-similarity search over a knowledge base
//! - **EchoTool**: passthrough for testing
//!
//! Custom tools implement the `Tool` trait and are registered with the agent.

use std::collections::HashMap;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// Tool trait
// ═══════════════════════════════════════════════════════════════════════════

/// A request from the agent to a tool.
#[derive(Clone, Debug)]
pub struct ToolRequest {
    /// Name of the tool to invoke.
    pub tool_name: String,
    /// Query embedding from the agent's hidden state, shape (dim,).
    pub query_embedding: Vec<f32>,
    /// Free-form text query (decoded from the agent's state).
    pub query_text: String,
    /// Recursion depth for tools that support it.
    pub depth: u8,
    /// Arbitrary key-value parameters.
    pub params: HashMap<String, String>,
}

/// A result returned by a tool.
#[derive(Clone, Debug)]
pub struct ToolResult {
    /// Name of the tool that produced this result.
    pub tool_name: String,
    /// Knowledge embedding to inject back into the agent's state, shape (dim,).
    pub knowledge_embedding: Vec<f32>,
    /// Human-readable summary of what the tool found.
    pub summary: String,
    /// Whether the tool succeeded.
    pub success: bool,
}

impl ToolResult {
    /// Create a failed result.
    pub fn failure(tool_name: &str, reason: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            knowledge_embedding: Vec::new(),
            summary: reason.to_string(),
            success: false,
        }
    }
}

/// Trait for agent tools.
///
/// Tools are the "slow-thinking" component of the SSD pattern.
/// They receive a structured request and return knowledge that gets
/// injected back into the Mamba-3 hidden state.
pub trait Tool: Send + Sync {
    /// Unique name of this tool.
    fn name(&self) -> &str;

    /// Short description for tool selection.
    fn description(&self) -> &str;

    /// The dimensionality of knowledge embeddings this tool returns.
    fn knowledge_dim(&self) -> usize;

    /// Execute the tool and return a result.
    fn execute(&self, request: &ToolRequest) -> ToolResult;
}

impl fmt::Debug for dyn Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tool({})", self.name())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ToolRegistry
// ═══════════════════════════════════════════════════════════════════════════

/// Registry of available tools for the agent.
///
/// Tool selection uses cosine similarity between the query embedding
/// and each tool's signature embedding. The signature is derived from
/// the tool's `knowledge_dim` and description hash, producing a
/// deterministic fingerprint that can be matched against query state.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    /// Cached signature embeddings per tool, shape (n_tools, max_dim).
    signatures: Vec<Vec<f32>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { tools: Vec::new(), signatures: Vec::new() }
    }

    /// Register a tool. Computes and caches its signature embedding.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let sig = Self::compute_signature(&*tool);
        self.signatures.push(sig);
        self.tools.push(tool);
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| &**t)
    }

    /// List all registered tool names.
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Select the best tool for a query using cosine similarity.
    ///
    /// Compares the query embedding against each tool's cached signature
    /// embedding and returns the tool with highest similarity.
    pub fn select_tool(&self, query_embedding: &[f32]) -> Option<&dyn Tool> {
        if self.tools.is_empty() {
            return None;
        }
        if self.tools.len() == 1 {
            return Some(&*self.tools[0]);
        }

        let mut best_sim = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, sig) in self.signatures.iter().enumerate() {
            let sim = cosine_sim(query_embedding, sig);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        Some(&*self.tools[best_idx])
    }

    /// Compute a deterministic signature embedding for a tool.
    ///
    /// Uses a hash of the tool's name and description to seed a
    /// pseudo-random embedding in the tool's knowledge_dim space.
    /// This gives each tool a unique, stable fingerprint that the
    /// query embedding can be compared against.
    fn compute_signature(tool: &dyn Tool) -> Vec<f32> {
        let dim = tool.knowledge_dim().max(1);
        let mut sig = vec![0.0f32; dim];

        // Simple deterministic hash → embedding
        let name = tool.name();
        let desc = tool.description();
        let mut hash: u64 = 5381;
        for b in name.bytes().chain(desc.bytes()) {
            hash = hash.wrapping_mul(33).wrapping_add(b as u64);
        }

        for d in 0..dim {
            // LCG-style deterministic pseudo-random from hash
            hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            sig[d] = ((hash >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }

        // L2-normalize the signature
        let norm: f32 = sig.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in &mut sig {
                *v /= norm;
            }
        }

        sig
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in: MemorySearchTool
// ═══════════════════════════════════════════════════════════════════════════

/// Vector-similarity memory search tool.
///
/// Stores (key, value) pairs and returns the nearest value for a query.
/// This is the "Recursive Deep Dive" from the SSD pattern — the agent
/// pauses its stream of consciousness to look something up.
pub struct MemorySearchTool {
    entries: Vec<(Vec<f32>, Vec<f32>, String)>, // (key, value, label)
    knowledge_dim: usize,
}

impl MemorySearchTool {
    /// Create a new memory search tool.
    pub fn new(knowledge_dim: usize) -> Self {
        Self { entries: Vec::new(), knowledge_dim }
    }

    /// Add a knowledge entry.
    pub fn add_entry(&mut self, key: Vec<f32>, value: Vec<f32>, label: String) {
        assert_eq!(key.len(), self.knowledge_dim);
        assert_eq!(value.len(), self.knowledge_dim);
        self.entries.push((key, value, label));
    }

    /// Number of entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

impl Tool for MemorySearchTool {
    fn name(&self) -> &str { "memory_search" }

    fn description(&self) -> &str {
        "Search vector knowledge base for relevant information"
    }

    fn knowledge_dim(&self) -> usize { self.knowledge_dim }

    fn execute(&self, request: &ToolRequest) -> ToolResult {
        if self.entries.is_empty() {
            return ToolResult::failure(self.name(), "Knowledge base is empty");
        }

        let query = &request.query_embedding;
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, (key, _, _)) in self.entries.iter().enumerate() {
            let sim = cosine_sim(query, key);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        let (_, ref value, ref label) = self.entries[best_idx];

        // Recursive refinement if depth > 0
        let final_value = if request.depth > 0 && self.entries.len() > 1 {
            // Blend query with result, re-search
            let dim = self.knowledge_dim.min(query.len());
            let mut refined = vec![0.0f32; self.knowledge_dim];
            for d in 0..dim {
                refined[d] = 0.5 * query[d] + 0.5 * value[d];
            }
            // Find second-nearest
            let mut best2_sim = f32::NEG_INFINITY;
            let mut best2_idx = 0;
            for (i, (key, _, _)) in self.entries.iter().enumerate() {
                let sim = cosine_sim(&refined, key);
                if sim > best2_sim {
                    best2_sim = sim;
                    best2_idx = i;
                }
            }
            self.entries[best2_idx].1.clone()
        } else {
            value.clone()
        };

        ToolResult {
            tool_name: self.name().to_string(),
            knowledge_embedding: final_value,
            summary: format!("Found: {} (similarity: {:.4})", label, best_sim),
            success: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in: EchoTool (for testing)
// ═══════════════════════════════════════════════════════════════════════════

/// Echo tool — returns the query embedding as-is.
/// Useful for testing the agent loop without external dependencies.
pub struct EchoTool {
    knowledge_dim: usize,
}

impl EchoTool {
    pub fn new(knowledge_dim: usize) -> Self {
        Self { knowledge_dim }
    }
}

impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "Echo query back (testing)" }
    fn knowledge_dim(&self) -> usize { self.knowledge_dim }

    fn execute(&self, request: &ToolRequest) -> ToolResult {
        let mut embedding = vec![0.0f32; self.knowledge_dim];
        let copy_len = self.knowledge_dim.min(request.query_embedding.len());
        embedding[..copy_len].copy_from_slice(&request.query_embedding[..copy_len]);

        ToolResult {
            tool_name: self.name().to_string(),
            knowledge_embedding: embedding,
            summary: format!("Echo: {}", request.query_text),
            success: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());

        registry.register(Box::new(EchoTool::new(4)));
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.names(), vec!["echo"]);
        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_echo_tool() {
        let tool = EchoTool::new(4);
        let request = ToolRequest {
            tool_name: "echo".into(),
            query_embedding: vec![1.0, 2.0, 3.0, 4.0],
            query_text: "hello".into(),
            depth: 0,
            params: HashMap::new(),
        };

        let result = tool.execute(&request);
        assert!(result.success);
        assert_eq!(result.knowledge_embedding, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_memory_search_tool() {
        let mut tool = MemorySearchTool::new(4);
        tool.add_entry(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![10.0, 20.0, 30.0, 40.0],
            "entry_a".into(),
        );
        tool.add_entry(
            vec![0.0, 1.0, 0.0, 0.0],
            vec![50.0, 60.0, 70.0, 80.0],
            "entry_b".into(),
        );

        let request = ToolRequest {
            tool_name: "memory_search".into(),
            query_embedding: vec![0.9, 0.1, 0.0, 0.0],
            query_text: "find A".into(),
            depth: 0,
            params: HashMap::new(),
        };

        let result = tool.execute(&request);
        assert!(result.success);
        assert_eq!(result.knowledge_embedding, vec![10.0, 20.0, 30.0, 40.0]);
        assert!(result.summary.contains("entry_a"));
    }

    #[test]
    fn test_memory_search_empty() {
        let tool = MemorySearchTool::new(4);
        let request = ToolRequest {
            tool_name: "memory_search".into(),
            query_embedding: vec![1.0, 0.0, 0.0, 0.0],
            query_text: "find".into(),
            depth: 0,
            params: HashMap::new(),
        };

        let result = tool.execute(&request);
        assert!(!result.success);
    }

    #[test]
    fn test_memory_search_recursive() {
        let mut tool = MemorySearchTool::new(4);
        tool.add_entry(vec![1.0, 0.0, 0.0, 0.0], vec![0.5, 0.5, 0.0, 0.0], "a".into());
        tool.add_entry(vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0], "b".into());

        let request = ToolRequest {
            tool_name: "memory_search".into(),
            query_embedding: vec![1.0, 0.0, 0.0, 0.0],
            query_text: "find".into(),
            depth: 1,
            params: HashMap::new(),
        };

        let result = tool.execute(&request);
        assert!(result.success);
        assert_eq!(result.knowledge_embedding.len(), 4);
    }

    #[test]
    fn test_tool_result_failure() {
        let result = ToolResult::failure("test", "something went wrong");
        assert!(!result.success);
        assert_eq!(result.tool_name, "test");
    }
}
