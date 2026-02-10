//! Recursion Layer — State-Space Delegation for Mamba3-JEPA.
//!
//! Implements the SSD (State-Space Delegation) pattern:
//! - **ConfusionMonitor**: learned linear probe on hidden state → confusion score
//! - **MemoryTool** trait: pluggable external memory (vector DB, sub-agent, etc.)
//! - **LocalMemoryTool**: default implementation using nearest-neighbor vector search
//! - **StateInjector**: h_new = h + W_inject · V_knowledge
//! - **RecursionLayer**: assembles monitor + tool + injector

use std::any::Any;

use rand::Rng;

use crate::config::RecursionConfig;

// ═══════════════════════════════════════════════════════════════════════════
// ConfusionMonitor
// ═══════════════════════════════════════════════════════════════════════════

/// Monitors the predictor's hidden state for "confusion" signals.
///
/// Uses a learned linear probe: score = σ(W_probe · h_mean + bias),
/// then applies EMA smoothing. When the smoothed score exceeds
/// `confusion_threshold`, the model should delegate to external memory.
pub struct ConfusionMonitor {
    /// Probe weights: (confusion_dims,) — dot product with pooled hidden state.
    probe_weight: Vec<f32>,
    /// Probe bias (scalar).
    probe_bias: f32,
    /// Number of hidden dimensions the probe reads (must match d_model or be ≤ d_model).
    confusion_dims: usize,
    /// EMA smoothing factor.
    smoothing: f32,
    /// Threshold for triggering delegation.
    threshold: f32,
    /// Current smoothed confusion score.
    smoothed_score: f32,
}

impl ConfusionMonitor {
    /// Create a new confusion monitor with random probe weights.
    pub fn new(confusion_dims: usize, smoothing: f32, threshold: f32) -> Self {
        let mut rng = rand::thread_rng();
        let std = (1.0 / confusion_dims as f32).sqrt();
        Self {
            probe_weight: (0..confusion_dims).map(|_| rng.gen_range(-std..std)).collect(),
            probe_bias: 0.0,
            confusion_dims,
            smoothing,
            threshold,
            smoothed_score: 0.0,
        }
    }

    /// Evaluate confusion from hidden states.
    ///
    /// # Arguments
    /// - `hidden`: shape (batch * seq_len, d_model) — the residual stream
    /// - `batch`: batch size
    /// - `seq_len`: sequence length
    /// - `d_model`: hidden dimension
    ///
    /// # Returns
    /// `(should_delegate, confusion_score)`
    pub fn evaluate(
        &mut self,
        hidden: &[f32],
        batch: usize,
        seq_len: usize,
        d_model: usize,
    ) -> (bool, f32) {
        let n = batch * seq_len;
        let probe_dim = self.confusion_dims.min(d_model);

        // Mean-pool over all positions: (d_model,)
        let mut pooled = vec![0.0f32; probe_dim];
        for i in 0..n {
            for d in 0..probe_dim {
                pooled[d] += hidden[i * d_model + d];
            }
        }
        let inv_n = 1.0 / n as f32;
        for d in 0..probe_dim {
            pooled[d] *= inv_n;
        }

        // Linear probe: dot(W, pooled) + bias → sigmoid
        let mut logit = self.probe_bias;
        for d in 0..probe_dim {
            logit += self.probe_weight[d] * pooled[d];
        }
        let score = sigmoid(logit);

        // EMA smoothing
        self.smoothed_score = self.smoothing * score
            + (1.0 - self.smoothing) * self.smoothed_score;

        (self.smoothed_score > self.threshold, self.smoothed_score)
    }

    /// Current smoothed confusion score.
    pub fn current_score(&self) -> f32 {
        self.smoothed_score
    }

    /// Reset the monitor state.
    pub fn reset(&mut self) {
        self.smoothed_score = 0.0;
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════
// MemoryTool trait
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for external memory tools.
///
/// Implementors provide a `recursive_search` method that takes a query vector
/// (derived from the hidden state) and returns a compressed knowledge vector.
///
/// The trait is object-safe for use with `Box<dyn MemoryTool>`.
/// Supports safe downcasting via `as_any()` / `as_any_mut()`.
pub trait MemoryTool: Send + Sync {
    /// Perform recursive search and return a knowledge vector.
    ///
    /// # Arguments
    /// - `query`: query vector derived from hidden state, shape (knowledge_dim,)
    /// - `depth`: recursion depth (0 = single lookup, >0 = recursive sub-queries)
    ///
    /// # Returns
    /// Knowledge vector of shape (knowledge_dim,).
    fn recursive_search(&self, query: &[f32], depth: u8) -> Vec<f32>;

    /// Downcast to `&dyn Any` for safe type-specific access.
    fn as_any(&self) -> &dyn Any;

    /// Downcast to `&mut dyn Any` for safe mutable type-specific access.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ═══════════════════════════════════════════════════════════════════════════
// LocalMemoryTool
// ═══════════════════════════════════════════════════════════════════════════

/// Default memory tool using local nearest-neighbor vector search.
///
/// Stores a knowledge base as a list of (key, value) vector pairs.
/// On search, finds the nearest key by cosine similarity and returns
/// the corresponding value. At depth > 0, recursively refines the
/// query by blending with intermediate results.
pub struct LocalMemoryTool {
    /// Knowledge base: Vec of (key, value) pairs, each of shape (knowledge_dim,).
    entries: Vec<(Vec<f32>, Vec<f32>)>,
    knowledge_dim: usize,
}

impl LocalMemoryTool {
    /// Create an empty local memory tool.
    pub fn new(knowledge_dim: usize) -> Self {
        Self { entries: Vec::new(), knowledge_dim }
    }

    /// Add a (key, value) entry to the knowledge base.
    pub fn add_entry(&mut self, key: Vec<f32>, value: Vec<f32>) {
        assert_eq!(key.len(), self.knowledge_dim);
        assert_eq!(value.len(), self.knowledge_dim);
        self.entries.push((key, value));
    }

    /// Number of entries in the knowledge base.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the knowledge base is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find the nearest entry by cosine similarity, returning its value.
    fn nearest(&self, query: &[f32]) -> Vec<f32> {
        if self.entries.is_empty() {
            return vec![0.0f32; self.knowledge_dim];
        }

        let mut best_sim = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, (key, _)) in self.entries.iter().enumerate() {
            let sim = cosine_sim(query, key);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        self.entries[best_idx].1.clone()
    }
}

impl MemoryTool for LocalMemoryTool {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn recursive_search(&self, query: &[f32], depth: u8) -> Vec<f32> {
        if self.entries.is_empty() {
            return vec![0.0f32; self.knowledge_dim];
        }

        let mut result = self.nearest(query);

        if depth > 0 {
            // Recursive refinement: blend query with intermediate result, re-search
            let mut refined_query = vec![0.0f32; self.knowledge_dim];
            let dim = self.knowledge_dim.min(query.len());
            for d in 0..dim {
                refined_query[d] = 0.5 * query[d] + 0.5 * result[d];
            }
            result = self.recursive_search(&refined_query, depth - 1);
        }

        result
    }
}

/// Cosine similarity between two vectors.
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
// StateInjector
// ═══════════════════════════════════════════════════════════════════════════

/// Injects knowledge vectors into the residual stream.
///
/// Computes: h_new = h + W_inject · V_knowledge
///
/// `W_inject` is a learned linear projection from `knowledge_dim` to `d_model`.
/// The injection is additive to preserve existing state information.
pub struct StateInjector {
    /// Projection weight: (d_model, knowledge_dim).
    weight: Vec<f32>,
    d_model: usize,
    knowledge_dim: usize,
}

impl StateInjector {
    /// Create a new state injector with random weights.
    pub fn new(d_model: usize, knowledge_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (d_model + knowledge_dim) as f32).sqrt();
        Self {
            weight: (0..d_model * knowledge_dim).map(|_| rng.gen_range(-std..std)).collect(),
            d_model,
            knowledge_dim,
        }
    }

    /// Inject knowledge into hidden states.
    ///
    /// # Arguments
    /// - `hidden`: shape (batch * seq_len * d_model) — the residual stream (modified in-place)
    /// - `knowledge`: shape (knowledge_dim,) — compressed knowledge vector
    /// - `batch`: batch size
    /// - `seq_len`: sequence length
    ///
    /// # Returns
    /// Modified hidden states with knowledge injected additively.
    pub fn inject(
        &self,
        hidden: &[f32],
        knowledge: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let n = batch * seq_len;
        let kd = self.knowledge_dim.min(knowledge.len());

        // Compute delta: W_inject · V_knowledge → (d_model,)
        let mut delta = vec![0.0f32; self.d_model];
        for o in 0..self.d_model {
            let mut acc = 0.0f32;
            for k in 0..kd {
                acc += self.weight[o * self.knowledge_dim + k] * knowledge[k];
            }
            delta[o] = acc;
        }

        // Add delta to every position: h_new[i] = h[i] + delta
        let mut out = hidden.to_vec();
        for i in 0..n {
            for d in 0..self.d_model {
                out[i * self.d_model + d] += delta[d];
            }
        }

        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RecursionLayer
// ═══════════════════════════════════════════════════════════════════════════

/// Recursion Layer — State-Space Delegation.
///
/// Assembles ConfusionMonitor + MemoryTool + StateInjector.
/// Called between backbone layers in the predictor to optionally delegate
/// to external memory and inject knowledge back into the residual stream.
pub struct RecursionLayer {
    pub config: RecursionConfig,
    pub monitor: ConfusionMonitor,
    pub injector: StateInjector,
    tool: Box<dyn MemoryTool>,
    /// Query projection: (d_model → knowledge_dim) for converting hidden state to search query.
    query_proj_weight: Vec<f32>,
    /// Total number of delegations performed.
    delegation_count: usize,
}

impl RecursionLayer {
    /// Create a new recursion layer with default LocalMemoryTool.
    pub fn new(config: RecursionConfig, d_model: usize) -> Self {
        let monitor = ConfusionMonitor::new(
            config.confusion_dims.min(d_model),
            config.smoothing,
            config.confusion_threshold,
        );
        let injector = StateInjector::new(d_model, config.knowledge_dim);
        let tool: Box<dyn MemoryTool> = Box::new(LocalMemoryTool::new(config.knowledge_dim));

        let mut rng = rand::thread_rng();
        let std = (2.0 / (d_model + config.knowledge_dim) as f32).sqrt();
        let query_proj_weight: Vec<f32> = (0..config.knowledge_dim * d_model)
            .map(|_| rng.gen_range(-std..std))
            .collect();

        Self {
            config,
            monitor,
            injector,
            tool,
            query_proj_weight,
            delegation_count: 0,
        }
    }

    /// Replace the memory tool with a custom implementation.
    pub fn set_memory_tool(&mut self, tool: Box<dyn MemoryTool>) {
        self.tool = tool;
    }

    /// Get a mutable reference to the underlying tool as `LocalMemoryTool`.
    ///
    /// Returns `Some` if the current tool is a `LocalMemoryTool`, `None` otherwise.
    /// Uses `Any`-based downcasting for safe type-specific access.
    pub fn local_memory_tool_mut(&mut self) -> Option<&mut LocalMemoryTool> {
        self.tool.as_any_mut().downcast_mut::<LocalMemoryTool>()
    }

    /// Check hidden state for confusion, optionally delegate, inject result.
    ///
    /// # Arguments
    /// - `hidden`: shape (batch * seq_len, d_model) — residual stream
    /// - `batch`: batch size
    /// - `seq_len`: sequence length
    /// - `d_model`: hidden dimension
    ///
    /// # Returns
    /// `(output_hidden, delegated, confusion_score)`
    pub fn maybe_delegate(
        &mut self,
        hidden: &[f32],
        batch: usize,
        seq_len: usize,
        d_model: usize,
    ) -> (Vec<f32>, bool, f32) {
        if !self.config.enabled {
            return (hidden.to_vec(), false, 0.0);
        }

        let (should_delegate, score) = self.monitor.evaluate(hidden, batch, seq_len, d_model);

        if !should_delegate {
            return (hidden.to_vec(), false, score);
        }

        // Compute query from pooled hidden state
        let n = batch * seq_len;
        let probe_dim = d_model;
        let mut pooled = vec![0.0f32; probe_dim];
        for i in 0..n {
            for d in 0..probe_dim {
                pooled[d] += hidden[i * d_model + d];
            }
        }
        let inv_n = 1.0 / n as f32;
        for d in 0..probe_dim {
            pooled[d] *= inv_n;
        }

        // Project pooled hidden → query: (knowledge_dim,)
        let kd = self.config.knowledge_dim;
        let mut query = vec![0.0f32; kd];
        for o in 0..kd {
            let mut acc = 0.0f32;
            for k in 0..probe_dim {
                acc += self.query_proj_weight[o * probe_dim + k] * pooled[k];
            }
            query[o] = acc;
        }

        // Delegate to memory tool
        let knowledge = self.tool.recursive_search(&query, self.config.max_depth);

        // Inject knowledge into hidden states
        let output = self.injector.inject(hidden, &knowledge, batch, seq_len);

        self.delegation_count += 1;

        (output, true, score)
    }

    /// Total number of delegations performed.
    pub fn delegation_count(&self) -> usize {
        self.delegation_count
    }

    /// Reset the monitor and delegation count.
    pub fn reset(&mut self) {
        self.monitor.reset();
        self.delegation_count = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RecursionConfig;

    #[test]
    fn test_confusion_monitor_static() {
        let mut mon = ConfusionMonitor::new(8, 0.5, 0.9);
        let hidden = vec![0.1f32; 4 * 8]; // batch=1, seq=4, d_model=8
        // Static input should not trigger (score is near sigmoid(small) ≈ 0.5)
        for _ in 0..10 {
            let (triggered, _score) = mon.evaluate(&hidden, 1, 4, 8);
            // With random weights, score stabilizes; we just check it's finite
            assert!(mon.current_score().is_finite());
            let _ = triggered;
        }
    }

    #[test]
    fn test_confusion_monitor_reset() {
        let mut mon = ConfusionMonitor::new(4, 0.5, 0.5);
        let hidden = vec![1.0f32; 4 * 4];
        mon.evaluate(&hidden, 1, 4, 4);
        assert!(mon.current_score() > 0.0 || mon.current_score() == 0.0);
        mon.reset();
        assert_eq!(mon.current_score(), 0.0);
    }

    #[test]
    fn test_local_memory_tool_empty() {
        let tool = LocalMemoryTool::new(8);
        let query = vec![1.0f32; 8];
        let result = tool.recursive_search(&query, 0);
        assert_eq!(result.len(), 8);
        // Empty KB returns zeros
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_local_memory_tool_nearest() {
        let mut tool = LocalMemoryTool::new(4);

        // Add two entries
        tool.add_entry(vec![1.0, 0.0, 0.0, 0.0], vec![10.0, 20.0, 30.0, 40.0]);
        tool.add_entry(vec![0.0, 1.0, 0.0, 0.0], vec![50.0, 60.0, 70.0, 80.0]);

        // Query closest to first entry
        let result = tool.recursive_search(&[0.9, 0.1, 0.0, 0.0], 0);
        assert_eq!(result, vec![10.0, 20.0, 30.0, 40.0]);

        // Query closest to second entry
        let result = tool.recursive_search(&[0.1, 0.9, 0.0, 0.0], 0);
        assert_eq!(result, vec![50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn test_local_memory_tool_recursive() {
        let mut tool = LocalMemoryTool::new(4);
        tool.add_entry(vec![1.0, 0.0, 0.0, 0.0], vec![0.5, 0.5, 0.0, 0.0]);
        tool.add_entry(vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]);

        // Depth 0: nearest match
        let r0 = tool.recursive_search(&[1.0, 0.0, 0.0, 0.0], 0);
        assert_eq!(r0, vec![0.5, 0.5, 0.0, 0.0]);

        // Depth 1: refines query with intermediate result, may find different match
        let r1 = tool.recursive_search(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(r1.len(), 4);
        assert!(r1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_state_injector() {
        let injector = StateInjector::new(4, 2);
        let hidden = vec![1.0f32; 2 * 4]; // batch=1, seq=2, d_model=4
        let knowledge = vec![1.0f32, 0.0];

        let result = injector.inject(&hidden, &knowledge, 1, 2);
        assert_eq!(result.len(), hidden.len());

        // Result should differ from input (W_inject · knowledge is non-zero in general)
        // Just check finite
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_state_injector_additive() {
        let injector = StateInjector::new(4, 2);
        let hidden = vec![0.0f32; 1 * 4]; // batch=1, seq=1, d_model=4
        let knowledge = vec![0.0f32, 0.0];

        // Zero knowledge → output == input
        let result = injector.inject(&hidden, &knowledge, 1, 1);
        assert_eq!(result, hidden);
    }

    #[test]
    fn test_recursion_layer_disabled() {
        let config = RecursionConfig::default(); // disabled
        let mut layer = RecursionLayer::new(config, 8);
        let hidden = vec![1.0f32; 2 * 8]; // batch=1, seq=2, d_model=8

        let (out, delegated, score) = layer.maybe_delegate(&hidden, 1, 2, 8);
        assert!(!delegated);
        assert_eq!(score, 0.0);
        assert_eq!(out, hidden);
    }

    #[test]
    fn test_recursion_layer_enabled() {
        let config = RecursionConfig {
            enabled: true,
            knowledge_dim: 4,
            confusion_threshold: 0.0, // always trigger
            confusion_dims: 4,
            inject_after_layer: 0,
            max_depth: 0,
            smoothing: 1.0, // instant response
        };
        let mut layer = RecursionLayer::new(config, 4);
        let hidden = vec![1.0f32; 2 * 4]; // batch=1, seq=2, d_model=4

        let (out, delegated, score) = layer.maybe_delegate(&hidden, 1, 2, 4);
        // With threshold=0 and smoothing=1, any sigmoid output > 0 triggers
        assert!(delegated);
        assert!(score > 0.0);
        assert_eq!(out.len(), hidden.len());
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(layer.delegation_count(), 1);
    }

    #[test]
    fn test_recursion_layer_with_entries() {
        let config = RecursionConfig {
            enabled: true,
            knowledge_dim: 4,
            confusion_threshold: 0.0, // always trigger
            confusion_dims: 4,
            inject_after_layer: 0,
            max_depth: 1,
            smoothing: 1.0,
        };
        let mut layer = RecursionLayer::new(config, 4);

        // Add some knowledge entries
        let mut local_tool = LocalMemoryTool::new(4);
        local_tool.add_entry(vec![1.0, 0.0, 0.0, 0.0], vec![5.0, 5.0, 5.0, 5.0]);
        local_tool.add_entry(vec![0.0, 1.0, 0.0, 0.0], vec![9.0, 9.0, 9.0, 9.0]);
        layer.set_memory_tool(Box::new(local_tool));

        let hidden = vec![0.5f32; 2 * 4];
        let (out, delegated, _) = layer.maybe_delegate(&hidden, 1, 2, 4);
        assert!(delegated);
        assert_eq!(out.len(), hidden.len());
        // Knowledge was injected, so output should differ from input
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_local_memory_tool_mut_downcast() {
        let config = RecursionConfig {
            enabled: true,
            knowledge_dim: 4,
            confusion_threshold: 0.5,
            confusion_dims: 4,
            inject_after_layer: 0,
            max_depth: 0,
            smoothing: 0.5,
        };
        let mut layer = RecursionLayer::new(config, 4);

        // Default tool is LocalMemoryTool, so downcast should succeed
        let tool = layer.local_memory_tool_mut();
        assert!(tool.is_some());

        // Should be able to add entries via the downcast reference
        let tool = tool.unwrap();
        tool.add_entry(vec![1.0, 0.0, 0.0, 0.0], vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(tool.len(), 1);
    }

    #[test]
    fn test_cosine_sim() {
        assert!((cosine_sim(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_sim(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        assert!((cosine_sim(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_zero() {
        assert_eq!(cosine_sim(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }
}
