use tch::Tensor;

pub fn comparison_out(out: &Tensor, candidates: &Tensor, query: &Tensor) -> Tensor {
    candidates
        .matmul_out(&out, &query)
        .sub_scalar_out(&out, 1.0)
        .div_scalar_out(&out, -2.0)
}

fn main() {
    let vectors: Vec<f32> = vec![
        0.0, 1.0, // 0
        1.0, 0.0, // 1
        -1.0, 0.0, // 2
        0.0, -1.0, // 3
        0.707, 0.707, // 4
        0.707, -0.707, // 5
        -0.707, 0.707, // 6
        -0.707, -0.707, // 7
    ];
    // neighborhood size  2
    let neighborhoods: Vec<i64> = vec![
        4, 6, // 0
        4, 5, // 1
        6, 7, // 2
        5, 7, // 3
        0, 1, // 4
        1, 3, // 5
        2, 0, // 6
        2, 5, // 7
    ];

    let vectors_t = Tensor::from_slice(&vectors);
    let vectors_t: Tensor = vectors_t.reshape(vec![8, 2]);
    vectors_t.print();
    let neighborhoods_t = Tensor::from_slice(&neighborhoods);
    let neighborhoods_t = neighborhoods_t.reshape(vec![8, 2]);
    neighborhoods_t.print();

    // query vector
    let q: Vec<f32> = vec![0.66436384, 0.74740932];
    let q_t = Tensor::from_slice(&q);
    let q_t = q_t.reshape(vec![2, 1]);

    q_t.print();
    let queue: Vec<i64> = vec![0];
    let queue_t = Tensor::from_slice(&queue);

    let candidates_t = vectors_t.index_select(0, &queue_t);
    eprintln!("candidates: ");
    candidates_t.print();

    eprintln!("------");
    let queue_distance_t: Tensor = (1.0 - candidates_t.mm(&q_t)) / 2.0;
    queue_distance_t.print();

    eprintln!("xxxxxxx");
    let out = queue_distance_t.zeros_like();
    let queue_distance_2_t: Tensor = comparison_out(&out, &candidates_t, &q_t);
    queue_distance_2_t.print();

    eprintln!("dimensions: {}", queue_distance_t.size()[0]);
    let k = i64::min(3, queue_distance_t.size()[0] as i64);
    let (vals, idxes) = queue_distance_t.topk(k, 0, false, true);
    idxes.print();
    vals.print();

    let best_idxs = idxes;
    let best_vals = vals;

    let neighbors = neighborhoods_t.index_select(0, &queue_t);
    neighbors.print();
}
