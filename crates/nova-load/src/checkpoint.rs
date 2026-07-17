use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("failed to read checkpoint `{path}`: {source}")]
    Read { path: String, source: std::io::Error },
    #[error("failed to parse checkpoint `{path}`: {source}")]
    Parse {
        path: String,
        source: serde_json::Error,
    },
    #[error("checkpoint `{path}` has unsupported version {found} (expected {expected})")]
    Version {
        path: String,
        found: u32,
        expected: u32,
    },
    #[error("checkpoint metadata mismatch: {0}")]
    Metadata(String),
    #[error("failed to write checkpoint `{path}`: {source}")]
    Write { path: String, source: std::io::Error },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointMeta {
    pub rank: usize,
    pub num_jobs: usize,
    pub datasource_identity: String,
    pub config_fingerprint: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckpointState {
    #[serde(default)]
    pub completed_files: BTreeSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointFile {
    version: u32,
    meta: CheckpointMeta,
    state: CheckpointState,
}

pub fn default_checkpoint_path(prefix: Option<&str>, meta: &CheckpointMeta) -> PathBuf {
    let root = prefix.unwrap_or(".nova-load-checkpoints");
    let short = short_fingerprint(&meta.config_fingerprint);
    PathBuf::from(root).join(format!(
        "load-{short}-rank{}-of{}.json",
        meta.rank, meta.num_jobs
    ))
}

pub fn scoped_path(path: &Path, rank: usize, num_jobs: usize) -> PathBuf {
    if num_jobs <= 1 {
        return path.to_path_buf();
    }
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "checkpoint".to_string());
    let ext = path.extension().map(|s| s.to_string_lossy().into_owned());
    let mut out = path.to_path_buf();
    let scoped = match ext {
        Some(ext) if !ext.is_empty() => format!("{stem}.rank{rank}of{num_jobs}.{ext}"),
        _ => format!("{stem}.rank{rank}of{num_jobs}"),
    };
    out.set_file_name(scoped);
    out
}

pub fn load_for_run(
    path: &Path,
    expected: &CheckpointMeta,
    resume: bool,
) -> Result<CheckpointState, CheckpointError> {
    if !resume {
        return Ok(CheckpointState::default());
    }
    if !path.exists() {
        return Ok(CheckpointState::default());
    }

    let raw = std::fs::read(path).map_err(|source| CheckpointError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let file: CheckpointFile =
        serde_json::from_slice(&raw).map_err(|source| CheckpointError::Parse {
            path: path.display().to_string(),
            source,
        })?;

    if file.version != CHECKPOINT_VERSION {
        return Err(CheckpointError::Version {
            path: path.display().to_string(),
            found: file.version,
            expected: CHECKPOINT_VERSION,
        });
    }
    validate_meta(path, &file.meta, expected)?;
    Ok(file.state)
}

pub fn save(
    path: &Path,
    meta: &CheckpointMeta,
    state: &CheckpointState,
) -> Result<(), CheckpointError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|source| CheckpointError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }

    let blob = CheckpointFile {
        version: CHECKPOINT_VERSION,
        meta: meta.clone(),
        state: state.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&blob).map_err(|source| CheckpointError::Parse {
        path: path.display().to_string(),
        source,
    })?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|source| CheckpointError::Write {
        path: tmp.display().to_string(),
        source,
    })?;
    std::fs::rename(&tmp, path).map_err(|source| CheckpointError::Write {
        path: path.display().to_string(),
        source,
    })?;
    Ok(())
}

fn validate_meta(
    path: &Path,
    found: &CheckpointMeta,
    expected: &CheckpointMeta,
) -> Result<(), CheckpointError> {
    if found.rank != expected.rank || found.num_jobs != expected.num_jobs {
        return Err(CheckpointError::Metadata(format!(
            "partition mismatch in `{}`: checkpoint is rank {}/{} but run is rank {}/{}",
            path.display(),
            found.rank,
            found.num_jobs,
            expected.rank,
            expected.num_jobs
        )));
    }
    if found.datasource_identity != expected.datasource_identity {
        return Err(CheckpointError::Metadata(format!(
            "datasource mismatch in `{}`: checkpoint datasource changed",
            path.display()
        )));
    }
    if found.config_fingerprint != expected.config_fingerprint {
        return Err(CheckpointError::Metadata(format!(
            "config mismatch in `{}`: fingerprint changed",
            path.display()
        )));
    }
    Ok(())
}

fn short_fingerprint(fingerprint: &str) -> String {
    let n = 12usize.min(fingerprint.len());
    fingerprint[..n].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> CheckpointMeta {
        CheckpointMeta {
            rank: 2,
            num_jobs: 7,
            datasource_identity: "s3://bucket/prefix".to_string(),
            config_fingerprint: "abcdef1234567890".to_string(),
        }
    }

    #[test]
    fn checkpoint_roundtrip_resume() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("state.json");
        let mut state = CheckpointState::default();
        state.completed_files.insert("a.parquet".to_string());
        state.completed_files.insert("b.parquet".to_string());
        save(&path, &meta(), &state).expect("save succeeds");

        let loaded = load_for_run(&path, &meta(), true).expect("load succeeds");
        assert_eq!(loaded.completed_files, state.completed_files);
    }

    #[test]
    fn resume_rejects_partition_mismatch() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("state.json");
        save(&path, &meta(), &CheckpointState::default()).expect("save succeeds");

        let mut expected = meta();
        expected.rank = 3;
        let err = load_for_run(&path, &expected, true).expect_err("must fail");
        assert!(matches!(err, CheckpointError::Metadata(msg) if msg.contains("partition mismatch")));
    }

    #[test]
    fn no_resume_ignores_existing_checkpoint() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("state.json");
        let mut state = CheckpointState::default();
        state.completed_files.insert("done.parquet".to_string());
        save(&path, &meta(), &state).expect("save succeeds");

        let loaded = load_for_run(&path, &meta(), false).expect("load succeeds");
        assert!(loaded.completed_files.is_empty());
    }
}
