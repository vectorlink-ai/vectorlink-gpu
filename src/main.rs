use queue::Queue;
use tch::{Kind, Tensor};
mod queue;

pub fn comparison(candidates: &Tensor, query: &Tensor) -> Tensor {
    (candidates.matmul(&query) - 1.0) / 2.0
}

#[allow(dead_code)]
pub struct SearchParams {
    circulant_parameter_count: usize,
    parallel_visit_count: usize,
    visit_queue_size: usize,
    search_queue_size: usize,
}

// Returns the number of candidates we are searching simultaneously.
pub fn search_from_seeds(
    query_vec: &Tensor,
    neighbor_indices: &Tensor,
    neighborhoods: &Tensor,
    vectors: &Tensor,
    _search_params: &SearchParams,
) -> (Tensor, Tensor) {
    // TODO primes
    let neighborhoods = neighborhoods.index_select(0, neighbor_indices);
    let neighborhood_vectors = vectors.index_select(0, &neighborhoods);
    let distances = comparison(&neighborhood_vectors, query_vec);
    let flat_neighbors = neighborhoods.flatten(0, -1);
    let flat_distances = distances.flatten(0, -1);
    (flat_neighbors, flat_distances)
}

pub fn closest_vectors(
    query_vec: &Tensor,
    search_queue: &mut Queue,
    vectors: &Tensor,
    neighborhoods: &Tensor,
    search_params: &SearchParams,
) {
    let [_, neighborhood_size] = neighborhoods.size().try_into().unwrap();
    let extra_capacity = neighborhood_size * search_params.parallel_visit_count as i64;
    let capacity = search_params.visit_queue_size as i64 + extra_capacity;
    let mut visit_queue = Queue::new_from(
        search_params.visit_queue_size as i64,
        capacity,
        search_queue,
    );
    let mut did_something = true;
    let mut seen = Tensor::empty([0], (Kind::Int, vectors.device()));
    while did_something {
        let neighbors_to_visit = visit_queue.pop_n_ids(search_params.parallel_visit_count as i64);
        let (indices, distances) = search_from_seeds(
            query_vec,
            &neighbors_to_visit,
            neighborhoods,
            vectors,
            search_params,
        );

        let mask = Tensor::isin(&indices, &seen, false, true);
        let masked_indices = indices.masked_select(&mask);
        let masked_distances = distances.masked_select(&mask);

        visit_queue.insert(&masked_indices, &masked_distances);
        did_something = search_queue.insert(&indices, &distances);

        seen = Tensor::concat(&[seen, masked_indices], 0);
    }
}

pub fn search_layers(
    layers: &[Tensor],
    query_vec: &Tensor,
    search_queue: &mut Queue,
    vectors: &Tensor,
    search_params: &SearchParams,
) {
    for layer in layers.iter() {
        closest_vectors(query_vec, search_queue, vectors, layer, search_params);
    }
}

pub fn search_from_initial(
    layers: &[Tensor],
    query_vec: &Tensor,
    vectors: &Tensor,
    search_params: &SearchParams,
) -> Queue {
    let n = std::cmp::min(
        layers[0].size1().unwrap(),
        search_params.parallel_visit_count as i64,
    );
    let initial_indices = Tensor::arange(n, (Kind::Int, query_vec.device()));
    let initial_vectors = vectors.index_select(0, &initial_indices);
    let initial_distances = comparison(&initial_vectors, query_vec);

    // assumes that last neighborhood is the biggest
    let (_, last_neighborhood_size) = layers.last().unwrap().size2().unwrap();

    let extra_capacity = last_neighborhood_size * search_params.parallel_visit_count as i64;
    let capacity = search_params.search_queue_size as i64 + extra_capacity;

    let mut search_queue = Queue::new(search_params.search_queue_size as i64, capacity);
    search_queue.insert(&initial_indices, &initial_distances);

    search_layers(layers, query_vec, &mut search_queue, vectors, search_params);

    search_queue
}

fn main() {
    todo!();
}
