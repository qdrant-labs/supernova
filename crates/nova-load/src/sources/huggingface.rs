use serde::Deserialize;

use super::ReaderOptions;

#[derive(Debug, Deserialize)]
pub struct HuggingfaceConfig {
    pub repo_id: String,
    #[serde(default)]
    pub subdir: Option<String>,
    #[serde(default)]
    pub file_list: Option<Vec<String>>,

    #[serde(flatten)]
    pub reader: ReaderOptions,
}