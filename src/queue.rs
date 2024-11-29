use tch::{Device, Kind, Tensor};

struct Queue {
    length: i64,
    indices: Tensor,
    distances: Tensor,
}

pub fn device() -> Device {
    Device::Cpu
}

impl Queue {
    pub fn new(length: i64, capacity: i64) -> Queue {
        // This is annoying, but u32 scalars using Tensor::full appear not to work.
        let indices = Tensor::from_slice(&vec![i32::MAX; capacity as usize]);
        let distances = Tensor::from_slice(&vec![f32::MAX; capacity as usize]);
        Queue {
            length,
            indices,
            distances,
        }
    }

    pub fn insert(&mut self, vector_ids: &Tensor, distances: &Tensor) -> bool {
        let buffer = self.indices.narrow_copy(0, 0, self.length);
        let capacity = self.indices.size1().unwrap();
        let vector_ids_length = vector_ids.size1().unwrap();
        let distances_length = distances.size1().unwrap();
        assert_eq!(vector_ids_length, distances_length);
        assert!(vector_ids_length + self.length <= capacity);
        let insert_length = vector_ids.size1().unwrap();
        let mut indices_tail = self.indices.narrow(0, self.length, insert_length);
        let mut distances_tail = self.distances.narrow(0, self.length, insert_length);
        indices_tail.copy_(&vector_ids);
        distances_tail.copy_(&distances);

        // NOTE: Make all of this work on windows of the beginning!

        // sort indices first
        let (sorted_index_values, sorted_index_indices) = self.indices.sort(0, false);

        self.indices.copy_(&sorted_index_values);
        self.distances
            .copy_(&self.distances.index_select(0, &sorted_index_indices));

        // sort distances stably
        let (candidate_distances, sort_indices) = self.distances.sort_stable(true, 0, false);
        let candidate_indices = &self.indices.index_select(0, &sort_indices);

        // remove duplicates
        let (unique_indices, inverse, _counts) =
            candidate_indices.unique_consecutive(true, true, 0);
        let device = self.indices.device();
        let perm = Tensor::arange(inverse.size1().unwrap(), (Kind::Int64, device));
        let inverse = inverse.flip([0]);
        let perm = perm.flip([0]);
        let map = inverse
            .new_empty(unique_indices.size1().unwrap(), (Kind::Int64, device))
            .scatter_(0, &inverse, &perm);

        let new_distances = candidate_distances.index_select(0, &map);
        let new_size = new_distances.size1().unwrap();
        let did_something = buffer.narrow(0, 0, new_size) != unique_indices;
        if did_something {
            let mut indices_window = self.indices.narrow(0, 0, new_size);
            indices_window.copy_(&unique_indices);
            self.distances.narrow(0, 0, new_size).copy_(&new_distances);
        }

        self.indices
            .narrow(0, new_size, capacity - new_size)
            .fill_(i32::MAX as i64);
        self.distances
            .narrow(0, new_size, capacity - new_size)
            .fill_(f32::MAX as f64);

        did_something
    }

    pub fn len(&self) -> i32 {
        let mask = (self.indices.eq(i32::MAX as i64));
        let result = Vec::<i32>::try_from(mask.nonzero().flatten(0, 1)).unwrap();
        result[0]
    }

    pub fn print(&self) {
        let size = self.indices.size1().unwrap();
        eprint!("Queue[ ");
        let indices = Vec::<i32>::try_from(&self.indices).unwrap();
        let distances = Vec::<f32>::try_from(&self.distances).unwrap();

        let length = self.len();
        for i in 0..size as usize {
            if i == length as usize {
                eprint!(" | ");
            }
            let vid = indices[i];
            let d = distances[i];

            let pair_string = if vid == i32::MAX && d == f32::MAX {
                format!("(i32::MAX, f32::MAX)")
            } else {
                format!("({vid}, {d})")
            };
            if i == size as usize - 1 {
                eprint!("{pair_string}");
            } else {
                eprint!("{pair_string}, ");
            }
        }
        eprintln!(" ]");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vectors() -> Tensor {
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
        Tensor::from_slice(&vectors).reshape(vec![8, 2])
    }

    fn neighborhoods() -> Tensor {
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
        Tensor::from_slice(&neighborhoods).reshape(vec![8, 2])
    }

    #[test]
    fn example() {
        // complete example
        let vector_ids = Tensor::from_slice(&vec![3, 0, 1, 3, 5]);
        let distances = Tensor::from_slice(&vec![-3.0, -10.0, -1.0, -3.0, 5.0]);
        eprintln!("vector_ids");
        vector_ids.print();
        let (vector_ids, indices) = vector_ids.sort(0, false);
        let distances = distances.index_select(0, &indices);
        eprintln!("vector_ids");
        vector_ids.print();
        eprintln!("distances");
        distances.print();
        let (distances, indices) = distances.sort_stable(true, 0, false);
        let vector_ids = vector_ids.index_select(0, &indices);

        let (vector_ids, inverse, _counts) = vector_ids.unique_consecutive(true, false, 0);
        let perm = Tensor::arange(inverse.size1().unwrap(), (Kind::Int, device()));
        let inverse = inverse.flip([0]);
        let perm = perm.flip([0]);
        let map = inverse
            .new_empty(vector_ids.size1().unwrap(), (Kind::Int, device()))
            .scatter_(0, &inverse, &perm);
        eprintln!("map");
        map.print();

        let distances = distances.index_select(0, &map);
        eprintln!("vector_ids");
        vector_ids.print();
        eprintln!("distances");
        distances.print();
    }

    #[test]
    fn insert_twice() {
        let length = 10;
        let capacity = 24;
        let mut queue = Queue::new(length, capacity);
        let vector_ids = Tensor::from_slice::<i32>(&[3, 0, 1]);
        let distances = Tensor::from_slice::<f32>(&[0.5, 1.0, 0.8]);
        queue.insert(&vector_ids, &distances);
        queue.print();
        let vector_ids = Tensor::from_slice::<i32>(&[3, 4, 3]);
        let distances = Tensor::from_slice::<f32>(&[0.5001, 0.01, 0.5]);
        queue.insert(&vector_ids, &distances);
        queue.print();
        panic!();
    }

    #[test]
    fn insert_under_capacity() {
        let length = 10;
        let capacity = 24;
        let mut queue = Queue::new(length, capacity);
        let vector_ids = Tensor::from_slice::<i32>(&[3, 0, 1, 2, 3, 4, 3]);
        let mut distances = Tensor::from_slice::<f32>(&[0.5, 1.0, 0.8, 0.3, 0.5001, 0.01, 0.5]);
        eprintln!("distances");
        distances.print();
        queue.insert(&vector_ids, &distances);
        queue.print();
        eprintln!("len: {}", queue.len());
        panic!();
    }

    #[test]
    fn tensor_equality() {
        let vector_ids1 = Tensor::from_slice::<i32>(&[3, 0, 1]);
        let vector_ids2 = Tensor::from_slice::<i32>(&[3, 0, 1]);
        assert!(vector_ids1 == vector_ids2);
    }

    #[test]
    fn did_something() {
        let length = 6;
        let capacity = 12;
        let mut queue = Queue::new(length, capacity);
        let vector_ids = Tensor::from_slice::<i32>(&[3, 0, 1]);
        let distances = Tensor::from_slice::<f32>(&[0.5, 1.0, 0.8]);
        let did_something = queue.insert(&vector_ids, &distances);
        eprintln!("did_something: {did_something}");

        let vector_ids = Tensor::from_slice::<i32>(&[3, 0, 1]);

        let distances = Tensor::from_slice::<f32>(&[0.5, 1.0, 0.8]);
        let did_something = queue.insert(&vector_ids, &distances);
        assert!(!did_something);

        let vector_ids = Tensor::from_slice::<i32>(&[3, 2, 5]);
        let distances = Tensor::from_slice::<f32>(&[0.5, 3.0, 9.8]);
        let did_something = queue.insert(&vector_ids, &distances);
        assert!(did_something);
    }
}
