//! Work partitioning for distributed loads.
//!
//! Each worker independently calls [`partition`] over the *same* deterministically
//! ordered file list (sources sort by key), so the slices are disjoint and
//! together cover every file — with no coordination between workers.

use crate::sources::FileRef;

/// Identifies one worker in a fleet: which slice of the files it owns.
#[derive(Debug, Clone, Copy)]
pub struct Partition {
    /// This worker's index, in `[0, num_jobs)`.
    pub rank: usize,
    /// Total number of workers.
    pub num_jobs: usize,
}

impl Partition {
    /// A single, non-distributed run: rank 0 of 1 (gets every file).
    pub fn single() -> Self {
        Partition { rank: 0, num_jobs: 1 }
    }

    /// Validate the rank/num_jobs pair. `num_jobs` must be ≥ 1 and `rank` must
    /// be in `[0, num_jobs)`.
    pub fn validate(self) -> Result<Self, String> {
        if self.num_jobs == 0 {
            return Err("--num-jobs must be at least 1".into());
        }
        if self.rank >= self.num_jobs {
            return Err(format!(
                "--job-rank {} is out of range for --num-jobs {} (valid: 0..{})",
                self.rank,
                self.num_jobs,
                self.num_jobs
            ));
        }
        Ok(self)
    }
}

/// Assign this worker its files by **stride**: ranks `r, r+n, r+2n, …`.
///
/// Stride beats contiguous chunks because it interleaves — if file sizes drift,
/// each worker ends up with a mix of large and small files, so the imbalance
/// averages out. Requires the input to be in the same order on every worker
/// (sources guarantee this by sorting on `key`).
pub fn partition(files: &[FileRef], p: Partition) -> Vec<FileRef> {
    files
        .iter()
        .skip(p.rank)
        .step_by(p.num_jobs)
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(keys: &[&str]) -> Vec<FileRef> {
        keys.iter().map(|k| FileRef { key: k.to_string(), size: None }).collect()
    }

    fn keys(files: &[FileRef]) -> Vec<String> {
        files.iter().map(|f| f.key.clone()).collect()
    }

    #[test]
    fn single_worker_gets_everything() {
        let all = files(&["a", "b", "c"]);
        assert_eq!(keys(&partition(&all, Partition::single())), ["a", "b", "c"]);
    }

    #[test]
    fn stride_is_disjoint_and_complete() {
        let all = files(&["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]);
        let n = 3;
        let mut seen = Vec::new();
        for rank in 0..n {
            seen.extend(keys(&partition(&all, Partition { rank, num_jobs: n })));
        }
        seen.sort();
        // Every file assigned exactly once across all workers.
        assert_eq!(seen, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]);
    }

    #[test]
    fn stride_interleaves() {
        let all = files(&["0", "1", "2", "3", "4", "5"]);
        assert_eq!(keys(&partition(&all, Partition { rank: 0, num_jobs: 3 })), ["0", "3"]);
        assert_eq!(keys(&partition(&all, Partition { rank: 1, num_jobs: 3 })), ["1", "4"]);
        assert_eq!(keys(&partition(&all, Partition { rank: 2, num_jobs: 3 })), ["2", "5"]);
    }

    #[test]
    fn more_workers_than_files_some_idle() {
        let all = files(&["a", "b"]);
        assert_eq!(keys(&partition(&all, Partition { rank: 0, num_jobs: 4 })), ["a"]);
        assert_eq!(keys(&partition(&all, Partition { rank: 1, num_jobs: 4 })), ["b"]);
        assert!(partition(&all, Partition { rank: 2, num_jobs: 4 }).is_empty());
        assert!(partition(&all, Partition { rank: 3, num_jobs: 4 }).is_empty());
    }

    #[test]
    fn validate_rejects_bad_input() {
        assert!(Partition { rank: 0, num_jobs: 0 }.validate().is_err());
        assert!(Partition { rank: 3, num_jobs: 3 }.validate().is_err());
        assert!(Partition { rank: 2, num_jobs: 3 }.validate().is_ok());
    }
}