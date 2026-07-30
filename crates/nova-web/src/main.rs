mod static_files;

use std::{
    collections::HashMap,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use nova_load::{LoadRuntimeOptions, config::LoadConfig, plan::Partition};
use nova_storm::config::StormConfig;
use qdrant_client::{
    Qdrant,
    qdrant::{
        GetCollectionInfoResponse, QueryPointsBuilder, RetrievedPoint, ScoredPoint,
        ScrollPointsBuilder, Value as QdrantValue, VectorsOutput, value::Kind as QdrantValueKind,
        vector_output::Vector as VectorOutputKind, vectors_output::VectorsOptions,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::task::JoinHandle;
use tracing::info;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    jobs: JobStore,
}

#[derive(Clone)]
struct JobStore {
    jobs: Arc<Mutex<HashMap<String, JobRecord>>>,
    handles: Arc<Mutex<HashMap<String, JoinHandle<()>>>>,
}

impl JobStore {
    fn new() -> Self {
        Self {
            jobs: Arc::new(Mutex::new(HashMap::new())),
            handles: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn insert(&self, job: JobRecord) {
        self.jobs
            .lock()
            .expect("job mutex poisoned")
            .insert(job.id.clone(), job);
    }

    fn get(&self, id: &str) -> Option<JobRecord> {
        self.jobs
            .lock()
            .expect("job mutex poisoned")
            .get(id)
            .cloned()
    }

    fn list(&self) -> Vec<JobRecord> {
        let mut items: Vec<_> = self
            .jobs
            .lock()
            .expect("job mutex poisoned")
            .values()
            .cloned()
            .collect();
        items.sort_by_key(|j| j.created_at_ms);
        items
    }

    fn with_job<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut JobRecord),
    {
        if let Some(job) = self.jobs.lock().expect("job mutex poisoned").get_mut(id) {
            f(job);
        }
    }

    fn insert_handle(&self, id: String, handle: JoinHandle<()>) {
        self.handles
            .lock()
            .expect("handle mutex poisoned")
            .insert(id, handle);
    }

    fn abort(&self, id: &str) -> bool {
        let handle = self
            .handles
            .lock()
            .expect("handle mutex poisoned")
            .remove(id);
        if let Some(handle) = handle {
            handle.abort();
            self.with_job(id, |job| {
                if job.status == JobStatus::Pending || job.status == JobStatus::Running {
                    job.status = JobStatus::Cancelled;
                    job.finished_at_ms = Some(now_ms());
                    job.logs.push("job cancelled by user".to_string());
                }
            });
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum JobStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JobRecord {
    id: String,
    kind: String,
    status: JobStatus,
    created_at_ms: u128,
    started_at_ms: Option<u128>,
    finished_at_ms: Option<u128>,
    result: Option<serde_json::Value>,
    error: Option<String>,
    logs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ConfigInput {
    config_path: Option<String>,
    config_yaml: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LoadJobRequest {
    #[serde(flatten)]
    config: ConfigInput,
    num_jobs: Option<usize>,
    job_rank: Option<usize>,
    resume: Option<bool>,
    checkpoint_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StormRunRequest {
    #[serde(flatten)]
    config: ConfigInput,
}

#[derive(Debug, Deserialize)]
struct DistLoadRequest {
    #[serde(flatten)]
    config: ConfigInput,
    resources: Option<String>,
    num_jobs: Option<usize>,
    pool_name: Option<String>,
    dry_run: Option<bool>,
    finalize: Option<bool>,
    catalog_path: Option<String>,
    catalog_remote_dir: Option<String>,
    build_catalog_input: Option<String>,
    build_catalog_output: Option<String>,
    build_catalog_resume: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct DistStormRequest {
    #[serde(flatten)]
    config: ConfigInput,
    resources: Option<String>,
    num_jobs: Option<usize>,
    pool_name: Option<String>,
    dry_run: Option<bool>,
    stage_query_source: Option<String>,
    query_source_remote_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StormReportRequest {
    inputs: Vec<String>,
    output_json: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QdrantRandomQueryRequest {
    collection_name: String,
    limit: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct QdrantScrollRequest {
    collection_name: String,
    limit: Option<u32>,
}

#[derive(Debug, Serialize)]
struct JobAccepted {
    job_id: String,
}

#[derive(Debug, Serialize)]
struct JobLogResponse {
    logs: Vec<String>,
}

#[derive(thiserror::Error, Debug)]
enum ApiError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(json!({ "error": self.to_string() }))).into_response()
    }
}

#[derive(Clone, Copy)]
enum LoadAction {
    Run,
    Prepare,
    Load,
    Finalize,
    Reindex,
    Delete,
    Inspect,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nova_web=info".into()),
        )
        .init();

    let dist_dir =
        PathBuf::from(std::env::var("DIST_DIR").unwrap_or_else(|_| {
            "web/supernova-dashboard/dist/supernova-dashboard/browser".to_owned()
        }));
    let static_files = if dist_dir.exists() {
        Some(static_files::load(&dist_dir)?)
    } else {
        None
    };

    let state = AppState {
        jobs: JobStore::new(),
    };
    let app = build_app(state, static_files);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8080);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(addr = %addr, "nova-web listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn build_app(state: AppState, static_files: Option<static_files::StaticFileMap>) -> Router {
    let api = Router::new()
        .route("/health", get(health))
        .route("/api/v1/load/run", post(load_run))
        .route("/api/v1/load/prepare", post(load_prepare))
        .route("/api/v1/load/load", post(load_load))
        .route("/api/v1/load/finalize", post(load_finalize))
        .route("/api/v1/load/reindex", post(load_reindex))
        .route("/api/v1/load/delete", post(load_delete))
        .route("/api/v1/load/inspect", post(load_inspect))
        .route("/api/v1/dist/load", post(dist_load))
        .route("/api/v1/dist/storm", post(dist_storm))
        .route("/api/v1/storm/run", post(storm_run))
        .route("/api/v1/storm/report", post(storm_report))
        .route("/api/v1/qdrant/collections", get(qdrant_collections))
        .route("/api/v1/qdrant/random-query", post(qdrant_random_query))
        .route("/api/v1/qdrant/scroll", post(qdrant_scroll))
        .route("/api/v1/jobs", get(list_jobs))
        .route("/api/v1/jobs/{job_id}", get(get_job))
        .route("/api/v1/jobs/{job_id}/logs", get(get_job_logs))
        .route("/api/v1/jobs/{job_id}/cancel", post(cancel_job))
        .with_state(state.clone());

    if let Some(static_files) = static_files {
        let static_router = Router::new()
            .route("/", get(static_files::serve))
            .route("/{*path}", get(static_files::serve))
            .with_state(static_files);
        api.merge(static_router)
    } else {
        api
    }
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn load_run(
    State(state): State<AppState>,
    Json(req): Json<LoadJobRequest>,
) -> Result<impl IntoResponse, ApiError> {
    submit_load_job(state, "load_run", LoadAction::Run, req).await
}

async fn load_prepare(
    State(state): State<AppState>,
    Json(req): Json<LoadJobRequest>,
) -> Result<impl IntoResponse, ApiError> {
    submit_load_job(state, "load_prepare", LoadAction::Prepare, req).await
}

async fn load_load(
    State(state): State<AppState>,
    Json(req): Json<LoadJobRequest>,
) -> Result<impl IntoResponse, ApiError> {
    submit_load_job(state, "load_load", LoadAction::Load, req).await
}

async fn load_finalize(
    State(state): State<AppState>,
    Json(req): Json<LoadJobRequest>,
) -> Result<impl IntoResponse, ApiError> {
    submit_load_job(state, "load_finalize", LoadAction::Finalize, req).await
}

async fn load_reindex(
    State(state): State<AppState>,
    Json(req): Json<LoadJobRequest>,
) -> Result<impl IntoResponse, ApiError> {
    submit_load_job(state, "load_reindex", LoadAction::Reindex, req).await
}

async fn load_delete(
    State(state): State<AppState>,
    Json(req): Json<LoadJobRequest>,
) -> Result<impl IntoResponse, ApiError> {
    submit_load_job(state, "load_delete", LoadAction::Delete, req).await
}

async fn load_inspect(
    State(state): State<AppState>,
    Json(req): Json<LoadJobRequest>,
) -> Result<impl IntoResponse, ApiError> {
    submit_load_job(state, "load_inspect", LoadAction::Inspect, req).await
}

async fn submit_load_job(
    state: AppState,
    kind: &str,
    action: LoadAction,
    req: LoadJobRequest,
) -> Result<impl IntoResponse, ApiError> {
    let config = parse_load_config(&req.config)?;
    let job_id = spawn_job(state, kind.to_string(), move || {
        run_in_worker(async move {
            let result = match action {
                LoadAction::Run => nova_load::run(config).await.map(|_| json!({"ok": true})),
                LoadAction::Prepare => nova_load::prepare(config)
                    .await
                    .map(|_| json!({"ok": true})),
                LoadAction::Finalize => nova_load::finalize(config)
                    .await
                    .map(|_| json!({"ok": true})),
                LoadAction::Reindex => nova_load::reindex(config)
                    .await
                    .map(|_| json!({"ok": true})),
                LoadAction::Delete => nova_load::delete(config).await.map(|_| json!({"ok": true})),
                LoadAction::Load => {
                    let part = Partition {
                        rank: req.job_rank.unwrap_or(0),
                        num_jobs: req.num_jobs.unwrap_or(1),
                    }
                    .validate()
                    .map_err(|e| e.to_string())?;
                    let runtime = LoadRuntimeOptions {
                        resume: req.resume.unwrap_or(false),
                        checkpoint_path: req.checkpoint_path.map(PathBuf::from),
                    };
                    nova_load::load(config, part, runtime)
                        .await
                        .map(|_| json!({"ok": true}))
                }
                LoadAction::Inspect => {
                    let part = Partition {
                        rank: req.job_rank.unwrap_or(0),
                        num_jobs: req.num_jobs.unwrap_or(1),
                    }
                    .validate()
                    .map_err(|e| e.to_string())?;
                    nova_load::inspect(config, part)
                        .await
                        .map(|_| json!({"ok": true}))
                }
            };
            result.map_err(|e| e.to_string())
        })
    });
    Ok((StatusCode::ACCEPTED, Json(JobAccepted { job_id })))
}

async fn storm_run(
    State(state): State<AppState>,
    Json(req): Json<StormRunRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let config = parse_storm_config(&req.config)?;
    let job_id = spawn_job(state, "storm_run".to_string(), move || {
        run_in_worker(async move {
            let summary = nova_storm::run(config).await.map_err(|e| e.to_string())?;
            serde_json::to_value(summary).map_err(|e| e.to_string())
        })
    });
    Ok((StatusCode::ACCEPTED, Json(JobAccepted { job_id })))
}

async fn dist_load(
    State(state): State<AppState>,
    Json(req): Json<DistLoadRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let finalize = req.finalize.unwrap_or(false);
    if !finalize && req.num_jobs.is_none() {
        return Err(ApiError::BadRequest(
            "num_jobs is required unless finalize=true".to_string(),
        ));
    }
    if req.build_catalog_input.is_some() && req.build_catalog_output.is_none() {
        return Err(ApiError::BadRequest(
            "build_catalog_output is required when build_catalog_input is set".to_string(),
        ));
    }
    validate_dist_config_input(&req.config)?;
    let job_id = spawn_job(state, "dist_load".to_string(), move || {
        let (config_path, temp_path) = materialize_dist_config(&req.config, "dist-load")?;
        let mut args = vec!["dist".to_string(), "load".to_string(), config_path];
        if let Some(resources) = req.resources {
            args.push("--resources".to_string());
            args.push(resources);
        }
        if let Some(pool_name) = req.pool_name {
            args.push("--pool-name".to_string());
            args.push(pool_name);
        }
        if req.dry_run.unwrap_or(false) {
            args.push("--dry-run".to_string());
        }
        if finalize {
            args.push("--finalize".to_string());
        } else if let Some(num_jobs) = req.num_jobs {
            args.push("--num-jobs".to_string());
            args.push(num_jobs.to_string());
        }
        if let Some(catalog_path) = req.catalog_path {
            args.push("--catalog".to_string());
            args.push(catalog_path);
        }
        if let Some(remote_dir) = req.catalog_remote_dir {
            args.push("--catalog-remote-dir".to_string());
            args.push(remote_dir);
        }
        if let Some(build_input) = req.build_catalog_input {
            args.push("--build-catalog-input".to_string());
            args.push(build_input);
        }
        if let Some(build_output) = req.build_catalog_output {
            args.push("--build-catalog-output".to_string());
            args.push(build_output);
        }
        if req.build_catalog_resume.unwrap_or(false) {
            args.push("--build-catalog-resume".to_string());
        }
        let output = run_dist_command(args);
        cleanup_temp_file(temp_path);
        output
    });
    Ok((StatusCode::ACCEPTED, Json(JobAccepted { job_id })))
}

async fn dist_storm(
    State(state): State<AppState>,
    Json(req): Json<DistStormRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if req.num_jobs.is_none() {
        return Err(ApiError::BadRequest("num_jobs is required".to_string()));
    }
    validate_dist_config_input(&req.config)?;
    let job_id = spawn_job(state, "dist_storm".to_string(), move || {
        let (config_path, temp_path) = materialize_dist_config(&req.config, "dist-storm")?;
        let mut args = vec![
            "dist".to_string(),
            "storm".to_string(),
            config_path,
            "--num-jobs".to_string(),
            req.num_jobs.unwrap_or(1).to_string(),
        ];
        if let Some(resources) = req.resources {
            args.push("--resources".to_string());
            args.push(resources);
        }
        if let Some(pool_name) = req.pool_name {
            args.push("--pool-name".to_string());
            args.push(pool_name);
        }
        if req.dry_run.unwrap_or(false) {
            args.push("--dry-run".to_string());
        }
        if let Some(stage_query_source) = req.stage_query_source {
            args.push("--stage-query-source".to_string());
            args.push(stage_query_source);
        }
        if let Some(remote_dir) = req.query_source_remote_dir {
            args.push("--query-source-remote-dir".to_string());
            args.push(remote_dir);
        }
        let output = run_dist_command(args);
        cleanup_temp_file(temp_path);
        output
    });
    Ok((StatusCode::ACCEPTED, Json(JobAccepted { job_id })))
}

async fn storm_report(
    State(state): State<AppState>,
    Json(req): Json<StormReportRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if req.inputs.is_empty() {
        return Err(ApiError::BadRequest("inputs must not be empty".to_string()));
    }
    let inputs = req
        .inputs
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let output_json = req.output_json.map(PathBuf::from);
    let job_id = spawn_job(state, "storm_report".to_string(), move || {
        let report = nova_storm::report::build_report(&inputs).map_err(|e| e.to_string())?;
        if let Some(path) = output_json {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
            std::fs::write(path, bytes).map_err(|e| e.to_string())?;
        }
        serde_json::to_value(report).map_err(|e| e.to_string())
    });
    Ok((StatusCode::ACCEPTED, Json(JobAccepted { job_id })))
}

async fn list_jobs(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    Ok(Json(state.jobs.list()))
}

async fn get_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<impl IntoResponse, ApiError> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("job `{job_id}` not found")))?;
    Ok(Json(job))
}

async fn get_job_logs(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<impl IntoResponse, ApiError> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("job `{job_id}` not found")))?;
    Ok(Json(JobLogResponse { logs: job.logs }))
}

async fn cancel_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<impl IntoResponse, ApiError> {
    let found = state.jobs.abort(&job_id);
    if !found && state.jobs.get(&job_id).is_none() {
        return Err(ApiError::NotFound(format!("job `{job_id}` not found")));
    }
    Ok(Json(json!({ "job_id": job_id, "cancelled": found })))
}

async fn qdrant_collections() -> Result<impl IntoResponse, ApiError> {
    let client = qdrant_client_from_env()?;
    let collections = client
        .list_collections()
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    let names = collections
        .collections
        .into_iter()
        .map(|c| c.name)
        .collect::<Vec<_>>();
    Ok(Json(json!({ "collections": names })))
}

async fn qdrant_random_query(
    Json(req): Json<QdrantRandomQueryRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let client = qdrant_client_from_env()?;
    let collection_info = client
        .collection_info(&req.collection_name)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    let vector_size = infer_dense_vector_size(&collection_info)
        .ok_or_else(|| ApiError::BadRequest("could not infer vector size".to_string()))?;
    let limit = req.limit.unwrap_or(10);
    let query = random_unit_vector(vector_size);
    let result = client
        .query(
            QueryPointsBuilder::new(&req.collection_name)
                .query(query.clone())
                .limit(limit)
                .with_payload(true)
                .with_vectors(true),
        )
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    let matches = result
        .result
        .into_iter()
        .filter_map(scored_point_to_json)
        .collect::<Vec<_>>();
    Ok(Json(json!({
        "collection_name": req.collection_name,
        "query_vector": query,
        "points": matches
    })))
}

async fn qdrant_scroll(
    Json(req): Json<QdrantScrollRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let client = qdrant_client_from_env()?;
    let points = client
        .scroll(
            ScrollPointsBuilder::new(&req.collection_name)
                .limit(req.limit.unwrap_or(2000))
                .with_payload(true)
                .with_vectors(true),
        )
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    let items = points
        .result
        .into_iter()
        .filter_map(retrieved_point_to_json)
        .collect::<Vec<_>>();
    Ok(Json(json!({
        "collection_name": req.collection_name,
        "points": items
    })))
}

fn parse_load_config(input: &ConfigInput) -> Result<LoadConfig, ApiError> {
    match (&input.config_path, &input.config_yaml) {
        (Some(_), Some(_)) => Err(ApiError::BadRequest(
            "provide either config_path or config_yaml, not both".to_string(),
        )),
        (None, None) => Err(ApiError::BadRequest(
            "missing config source: set config_path or config_yaml".to_string(),
        )),
        (Some(path), None) => LoadConfig::from_path(path)
            .map_err(|e| ApiError::BadRequest(format!("load config parse failed: {e}"))),
        (None, Some(yaml)) => LoadConfig::from_yaml(yaml)
            .map_err(|e| ApiError::BadRequest(format!("load config parse failed: {e}"))),
    }
}

fn parse_storm_config(input: &ConfigInput) -> Result<StormConfig, ApiError> {
    match (&input.config_path, &input.config_yaml) {
        (Some(_), Some(_)) => Err(ApiError::BadRequest(
            "provide either config_path or config_yaml, not both".to_string(),
        )),
        (None, None) => Err(ApiError::BadRequest(
            "missing config source: set config_path or config_yaml".to_string(),
        )),
        (Some(path), None) => StormConfig::from_path(path)
            .map_err(|e| ApiError::BadRequest(format!("storm config parse failed: {e}"))),
        (None, Some(yaml)) => StormConfig::from_yaml(yaml)
            .map_err(|e| ApiError::BadRequest(format!("storm config parse failed: {e}"))),
    }
}

fn spawn_job<F>(state: AppState, kind: String, job_fn: F) -> String
where
    F: FnOnce() -> Result<serde_json::Value, String> + Send + 'static,
{
    let id = Uuid::new_v4().to_string();
    state.jobs.insert(JobRecord {
        id: id.clone(),
        kind,
        status: JobStatus::Pending,
        created_at_ms: now_ms(),
        started_at_ms: None,
        finished_at_ms: None,
        result: None,
        error: None,
        logs: vec!["job accepted".to_string()],
    });
    let jobs = state.jobs.clone();
    let run_id = id.clone();
    let handle = tokio::task::spawn_blocking(move || {
        jobs.with_job(&run_id, |job| {
            job.status = JobStatus::Running;
            job.started_at_ms = Some(now_ms());
            job.logs.push("job started".to_string());
        });
        match job_fn() {
            Ok(result) => jobs.with_job(&run_id, |job| {
                job.status = JobStatus::Succeeded;
                job.finished_at_ms = Some(now_ms());
                job.result = Some(result);
                job.logs.push("job finished successfully".to_string());
            }),
            Err(error) => jobs.with_job(&run_id, |job| {
                if job.status == JobStatus::Cancelled {
                    return;
                }
                job.status = JobStatus::Failed;
                job.finished_at_ms = Some(now_ms());
                job.error = Some(error);
                job.logs.push("job failed".to_string());
            }),
        }
    });
    state.jobs.insert_handle(id.clone(), handle);
    id
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn qdrant_client_from_env() -> Result<Qdrant, ApiError> {
    let url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());
    let api_key = std::env::var("QDRANT_API_KEY").ok();
    let mut builder = Qdrant::from_url(&url);
    if let Some(api_key) = api_key
        && !api_key.is_empty()
    {
        builder = builder.api_key(api_key);
    }
    builder
        .build()
        .map_err(|e| ApiError::Internal(format!("qdrant client error: {e}")))
}

fn infer_dense_vector_size(resp: &GetCollectionInfoResponse) -> Option<usize> {
    let cfg = resp.result.as_ref()?.config.as_ref()?;
    let vectors_cfg = cfg.params.as_ref()?.vectors_config.as_ref()?;
    match vectors_cfg.config.as_ref()? {
        qdrant_client::qdrant::vectors_config::Config::Params(p) => Some(p.size as usize),
        qdrant_client::qdrant::vectors_config::Config::ParamsMap(m) => {
            m.map.values().next().map(|v| v.size as usize)
        }
    }
}

fn random_unit_vector(size: usize) -> Vec<f32> {
    let mut values = (0..size)
        .map(|_| (rand::random::<f32>() * 2.0) - 1.0)
        .collect::<Vec<_>>();
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return random_unit_vector(size);
    }
    for v in &mut values {
        *v /= norm;
    }
    values
}

fn point_id_to_json(id: qdrant_client::qdrant::PointId) -> serde_json::Value {
    match id.point_id_options {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => json!(n),
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) => json!(u),
        None => serde_json::Value::Null,
    }
}

fn scored_point_to_json(point: ScoredPoint) -> Option<serde_json::Value> {
    let vector = extract_dense_vector(point.vectors?)?;
    Some(json!({
        "id": point.id.map(point_id_to_json).unwrap_or(serde_json::Value::Null),
        "score": point.score,
        "vector": vector,
        "payload": payload_to_json(point.payload),
    }))
}

fn retrieved_point_to_json(point: RetrievedPoint) -> Option<serde_json::Value> {
    let vector = extract_dense_vector(point.vectors?)?;
    Some(json!({
        "id": point.id.map(point_id_to_json).unwrap_or(serde_json::Value::Null),
        "vector": vector,
        "payload": payload_to_json(point.payload),
    }))
}

fn extract_dense_vector(vectors: VectorsOutput) -> Option<Vec<f32>> {
    match vectors.vectors_options? {
        VectorsOptions::Vector(v) => vector_output_to_dense(v),
        VectorsOptions::Vectors(named) => {
            named.vectors.into_values().find_map(vector_output_to_dense)
        }
    }
}

fn vector_output_to_dense(vector: qdrant_client::qdrant::VectorOutput) -> Option<Vec<f32>> {
    match vector.vector? {
        VectorOutputKind::Dense(dense) => Some(dense.data),
        VectorOutputKind::Sparse(_) => None,
        VectorOutputKind::MultiDense(multi) => multi.vectors.into_iter().next().map(|v| v.data),
    }
}

fn payload_to_json(payload: std::collections::HashMap<String, QdrantValue>) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (key, value) in payload {
        out.insert(key, qdrant_value_to_json(value));
    }
    serde_json::Value::Object(out)
}

fn qdrant_value_to_json(value: QdrantValue) -> serde_json::Value {
    match value.kind {
        Some(QdrantValueKind::NullValue(_)) => serde_json::Value::Null,
        Some(QdrantValueKind::DoubleValue(v)) => json!(v),
        Some(QdrantValueKind::IntegerValue(v)) => json!(v),
        Some(QdrantValueKind::StringValue(v)) => json!(v),
        Some(QdrantValueKind::BoolValue(v)) => json!(v),
        Some(QdrantValueKind::StructValue(v)) => {
            let mut out = serde_json::Map::new();
            for (k, inner) in v.fields {
                out.insert(k, qdrant_value_to_json(inner));
            }
            serde_json::Value::Object(out)
        }
        Some(QdrantValueKind::ListValue(v)) => serde_json::Value::Array(
            v.values
                .into_iter()
                .map(qdrant_value_to_json)
                .collect::<Vec<_>>(),
        ),
        None => serde_json::Value::Null,
    }
}

fn run_in_worker<F>(fut: F) -> Result<serde_json::Value, String>
where
    F: std::future::Future<Output = Result<serde_json::Value, String>>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(fut)
}

fn validate_dist_config_input(input: &ConfigInput) -> Result<(), ApiError> {
    match (&input.config_path, &input.config_yaml) {
        (Some(_), Some(_)) => Err(ApiError::BadRequest(
            "provide either config_path or config_yaml, not both".to_string(),
        )),
        (None, None) => Err(ApiError::BadRequest(
            "missing config source: set config_path or config_yaml".to_string(),
        )),
        _ => Ok(()),
    }
}

fn materialize_dist_config(
    input: &ConfigInput,
    label: &str,
) -> Result<(String, Option<PathBuf>), String> {
    match (&input.config_path, &input.config_yaml) {
        (Some(path), None) => Ok((path.clone(), None)),
        (None, Some(yaml)) => {
            let mut path = std::env::temp_dir();
            path.push(format!("nova-web-{label}-{}.yaml", Uuid::new_v4()));
            std::fs::write(&path, yaml).map_err(|e| format!("failed to write temp config: {e}"))?;
            Ok((path.to_string_lossy().to_string(), Some(path)))
        }
        (Some(_), Some(_)) => {
            Err("provide either config_path or config_yaml, not both".to_string())
        }
        (None, None) => Err("missing config source: set config_path or config_yaml".to_string()),
    }
}

fn cleanup_temp_file(path: Option<PathBuf>) {
    if let Some(path) = path {
        let _ = std::fs::remove_file(path);
    }
}

fn run_dist_command(args: Vec<String>) -> Result<serde_json::Value, String> {
    let binary = std::env::var("NOVA_DIST_BIN").unwrap_or_else(|_| "nova".to_string());
    let workspace = workspace_root()?;
    let output = Command::new(&binary)
        .args(&args)
        .current_dir(workspace)
        .output()
        .map_err(|e| format!("failed to start `{binary}`: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        return Err(format!(
            "`{binary} {}` failed with exit code {code}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            args.join(" ")
        ));
    }
    Ok(json!({
        "command": format!("{binary} {}", args.join(" ")),
        "status": output.status.code(),
        "stdout": stdout,
        "stderr": stderr,
    }))
}

fn workspace_root() -> Result<PathBuf, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    root.canonicalize()
        .map_err(|e| format!("failed to resolve workspace root: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, header},
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_endpoint_returns_ok() {
        let app = build_app(
            AppState {
                jobs: JobStore::new(),
            },
            None,
        );
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn spa_fallback_serves_index_html() {
        let tmp = tempfile::tempdir().expect("temp dir");
        std::fs::write(tmp.path().join("index.html"), "<html>supernova</html>")
            .expect("write index");
        let assets = static_files::load(tmp.path()).expect("load assets");
        let app = build_app(
            AppState {
                jobs: JobStore::new(),
            },
            Some(assets),
        );

        let res = app
            .oneshot(
                Request::builder()
                    .uri("/does-not-exist")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn dist_endpoint_is_registered() {
        let app = build_app(
            AppState {
                jobs: JobStore::new(),
            },
            None,
        );
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/dist/storm")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from("{}"))
            .expect("request");
        let res = app.oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }
}
