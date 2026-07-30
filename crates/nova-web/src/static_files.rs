use std::{collections::HashMap, path::Path, sync::Arc};

use axum::{
    body::Bytes,
    extract::State,
    http::{StatusCode, Uri, header},
    response::{IntoResponse, Response},
};
use walkdir::WalkDir;

pub struct StaticFile {
    pub bytes: Bytes,
    pub content_type: String,
}

#[derive(Clone)]
pub struct StaticFileMap(pub Arc<HashMap<String, StaticFile>>);

pub fn load(dir: &Path) -> anyhow::Result<StaticFileMap> {
    let mut map = HashMap::new();
    for entry in WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let abs = entry.path();
        let rel = abs.strip_prefix(dir)?;
        let url_path = format!(
            "/{}",
            rel.to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/")
        );
        let bytes = std::fs::read(abs)?;
        let content_type = mime_guess::from_path(abs)
            .first_or_octet_stream()
            .to_string();
        map.insert(
            url_path,
            StaticFile {
                bytes: Bytes::from(bytes),
                content_type,
            },
        );
    }
    tracing::info!(files = map.len(), dir = %dir.display(), "loaded static assets");
    Ok(StaticFileMap(Arc::new(map)))
}

pub async fn serve(State(map): State<StaticFileMap>, uri: Uri) -> Response {
    let path = uri.path();
    if let Some(file) = map.0.get(path) {
        return build_response(&file.bytes, &file.content_type);
    }
    // SPA fallback: support both CSR and SSR output layouts.
    if let Some(index) = map
        .0
        .get("/index.html")
        .or_else(|| map.0.get("/index.csr.html"))
    {
        return build_response(&index.bytes, &index.content_type);
    }
    (StatusCode::NOT_FOUND, "404 Not Found").into_response()
}

fn build_response(bytes: &Bytes, content_type: &str) -> Response {
    ([(header::CONTENT_TYPE, content_type)], bytes.clone()).into_response()
}
