//! InfoNCE contrastive loss for Mamba3-JEPA.

/// Bi-directional InfoNCE loss.
///
/// Computes cosine similarity between L2-normalized prediction and target embeddings,
/// then applies symmetric cross-entropy.
///
/// # Arguments
/// - `predictions`: shape (batch, embed_dim)
/// - `targets`: shape (batch, embed_dim)
/// - `temperature`: scaling factor for similarity logits
///
/// # Returns
/// Scalar loss value.
pub fn info_nce_loss(
    predictions: &[f32],
    targets: &[f32],
    batch: usize,
    embed_dim: usize,
    temperature: f32,
) -> f32 {
    assert_eq!(predictions.len(), batch * embed_dim);
    assert_eq!(targets.len(), batch * embed_dim);
    assert!(batch > 0);

    // L2-normalize predictions and targets
    let pred_norm = l2_normalize(predictions, batch, embed_dim);
    let tgt_norm = l2_normalize(targets, batch, embed_dim);

    // Cosine similarity matrix: (batch, batch)
    // sim[i][j] = dot(pred_norm[i], tgt_norm[j]) / temperature
    let mut sim = vec![0.0f32; batch * batch];
    for i in 0..batch {
        for j in 0..batch {
            let mut dot = 0.0f32;
            for k in 0..embed_dim {
                dot += pred_norm[i * embed_dim + k] * tgt_norm[j * embed_dim + k];
            }
            sim[i * batch + j] = dot / temperature;
        }
    }

    // Bi-directional cross-entropy: labels are the diagonal (i matches i)
    let loss_p2t = cross_entropy_rows(&sim, batch);
    let loss_t2p = cross_entropy_cols(&sim, batch);

    (loss_p2t + loss_t2p) / 2.0
}

/// L2-normalize each row of a (rows, dim) matrix.
fn l2_normalize(data: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let start = r * dim;
        let norm: f32 = data[start..start + dim]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        for d in 0..dim {
            out[start + d] = data[start + d] / norm;
        }
    }
    out
}

/// Cross-entropy along rows: -log(softmax(sim[i])[i]) averaged over i.
fn cross_entropy_rows(sim: &[f32], batch: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..batch {
        let row_start = i * batch;
        let max_val = sim[row_start..row_start + batch]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = sim[row_start..row_start + batch]
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum();
        let log_softmax = sim[row_start + i] - max_val - exp_sum.ln();
        total -= log_softmax;
    }
    total / batch as f32
}

/// Cross-entropy along columns: -log(softmax(sim[:, j])[j]) averaged over j.
fn cross_entropy_cols(sim: &[f32], batch: usize) -> f32 {
    let mut total = 0.0f32;
    for j in 0..batch {
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..batch {
            max_val = max_val.max(sim[i * batch + j]);
        }
        let exp_sum: f32 = (0..batch)
            .map(|i| (sim[i * batch + j] - max_val).exp())
            .sum();
        let log_softmax = sim[j * batch + j] - max_val - exp_sum.ln();
        total -= log_softmax;
    }
    total / batch as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_info_nce_matched_pairs() {
        // Each prediction matches its corresponding target (orthogonal across samples).
        // With high similarity on diagonal and low off-diagonal, loss should be low.
        let batch = 4;
        let dim = 8;
        let mut pred = vec![0.0f32; batch * dim];
        let mut tgt = vec![0.0f32; batch * dim];
        for b in 0..batch {
            // Each sample is a one-hot in a different dimension
            pred[b * dim + b] = 1.0;
            tgt[b * dim + b] = 1.0;
        }
        let loss = info_nce_loss(&pred, &tgt, batch, dim, 0.07);
        // Diagonal has sim=1/0.07≈14.3, off-diagonal has sim=0.
        // softmax(14.3, 0, 0, 0) ≈ 1.0, so loss ≈ 0
        assert!(loss < 0.01, "matched orthogonal pairs should give near-zero loss, got {}", loss);
    }

    #[test]
    fn test_info_nce_random() {
        // Random vectors: loss should be finite and positive
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let batch = 16;
        let dim = 32;
        let pred: Vec<f32> = (0..batch * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let tgt: Vec<f32> = (0..batch * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let loss = info_nce_loss(&pred, &tgt, batch, dim, 0.07);
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
        assert!(loss > 0.0, "loss should be positive, got {}", loss);
    }

    #[test]
    fn test_l2_normalize() {
        let data = vec![3.0, 4.0];
        let normed = l2_normalize(&data, 1, 2);
        assert!((normed[0] - 0.6).abs() < 1e-5);
        assert!((normed[1] - 0.8).abs() < 1e-5);
    }
}
