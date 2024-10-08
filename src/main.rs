// Embeddings -> Search
const N_DIMS: usize = 4;

// Move to f16
const EMBEDDINGS: [[f32; N_DIMS]; 5] = [
    [0.0, -1.0, 3.1, 0.4],
    [1.0, -1.0, 3.1, 0.4],
    [2.0, -1.0, 3.1, 0.4],
    [3.0, -1.0, 3.1, 0.4],
    [4.0, -1.0, 3.1, 0.4],
];

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}

fn main() {
    let query: [f32; N_DIMS] = [0.0, -0.5, 2.0, -1.0];

    for (i, embedding) in EMBEDDINGS.iter().enumerate() {
        let similarity = cosine_similarity(&query, embedding);
        println!("Embedding {}: Cosine Similarity = {}", i, similarity);
    }
}
