//! Milvus load backend (feature `milvus`, crate `milvus-sdk-rust`).
//!
//! Milvus inserts are **columnar** (`FieldColumn` per field), so `upsert_batch`
//! transposes the row-shaped [`Point`]s into one id column + one column per dense
//! vector. The primary key is a **varchar** (every id is stringified, so UUID
//! point ids work), and vectors are float-vector fields.
//!
//! Scope for now: **dense vectors, id only**. This SDK exposes no JSON/dynamic
//! field, so **payload is not persisted** — it's dropped with a warning. Sparse
//! and multivector values error out. The SDK also has no `COSINE` metric, so
//! cosine maps to inner-product (correct only for L2-normalized vectors).
//!
//! In this SDK the data-plane ops (`insert`/`flush`/`create_index`/`load`) live on
//! a `Collection` obtained via `client.get_collection(name)`, not on the client.
//!
//! Worker note: distributed `load` workers call `upsert_batch` without ever
//! calling `ensure_collection` (the controller does that in `prepare`). So the
//! insert path builds its `FieldSchema`s from the data itself (dim inferred from
//! the vectors) rather than a schema stashed at create time.

use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use milvus::client::{Client, ClientBuilder};
use milvus::data::FieldColumn;
use milvus::index::{IndexParams, IndexType, MetricType};
use milvus::schema::{CollectionSchemaBuilder, FieldSchema};
use serde::Deserialize;

use crate::config::VectorKind;
use crate::stores::{CollectionSchema, Point, PointId, StoreError, VectorStore, VectorValue};

const PK_FIELD: &str = "id";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MilvusConfig {
    /// e.g. `http://localhost:19530`.
    pub url: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    #[serde(default = "default_collection")]
    pub collection_name: String,
    /// Max length for the varchar primary key (UUIDs need 36).
    #[serde(default = "default_id_len")]
    pub id_max_length: i32,
    /// Drop + recreate the collection if it already exists.
    #[serde(default)]
    pub recreate: bool,
}

fn default_collection() -> String {
    "default".to_string()
}
fn default_id_len() -> i32 {
    128
}

pub struct MilvusStore {
    client: Client,
    collection_name: String,
    id_max_length: i32,
    recreate: bool,
}

impl MilvusConfig {
    pub async fn connect(self) -> Result<MilvusStore, StoreError> {
        // Owned String: tonic's Endpoint is TryFrom<String>, not From<&String>.
        let mut builder = ClientBuilder::new(self.url);
        if let Some(u) = &self.username {
            builder = builder.username(u);
        }
        if let Some(p) = &self.password {
            builder = builder.password(p);
        }
        let client = builder.build().await.map_err(to_other)?;
        Ok(MilvusStore {
            client,
            collection_name: self.collection_name,
            id_max_length: self.id_max_length,
            recreate: self.recreate,
        })
    }
}

fn to_other<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Other(e.to_string())
}

/// Qdrant-style distance name → Milvus metric. This SDK has no `COSINE`, so
/// cosine falls back to inner product (equivalent only for normalized vectors).
fn metric(distance: Option<&str>) -> MetricType {
    match distance.unwrap_or("cosine") {
        "euclid" => MetricType::L2,
        _ => MetricType::IP, // dot, and cosine (see note above)
    }
}

impl fmt::Display for MilvusStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "milvus({})", self.collection_name)
    }
}

/// Dense vector fields (name, dim), erroring on sparse/multivector.
fn dense_fields(schema: &CollectionSchema) -> Result<Vec<(&String, i64)>, StoreError> {
    let mut out = Vec::new();
    for (name, spec) in &schema.vectors {
        match spec.kind {
            VectorKind::Dense => {
                let dim = *schema.dims.get(name).ok_or_else(|| {
                    StoreError::Other(format!("vector `{name}`: dims not resolved"))
                })?;
                out.push((name, dim as i64));
            }
            VectorKind::Sparse | VectorKind::Multivector => {
                return Err(StoreError::Other(format!(
                    "milvus backend supports dense vectors only for now; vector `{name}` is {:?}",
                    spec.kind
                )));
            }
        }
    }
    Ok(out)
}

#[async_trait]
impl VectorStore for MilvusStore {
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        let name = self.collection_name.as_str();
        let dense = dense_fields(schema)?;

        if self.client.has_collection(name).await.map_err(to_other)? {
            if !self.recreate {
                return Ok(());
            }
            self.client.drop_collection(name).await.map_err(to_other)?;
        }

        tracing::warn!(
            "milvus backend does not persist payload (the SDK has no JSON field); \
             only the id + dense vectors are stored"
        );
        if schema
            .vectors
            .values()
            .any(|s| matches!(s.distance.as_deref(), None | Some("cosine")))
        {
            tracing::warn!(
                "milvus SDK has no COSINE metric — cosine vectors are indexed with \
                 inner product; ensure they are L2-normalized"
            );
        }

        // add_field takes &mut self (mutates in place), so don't reassign.
        let mut builder = CollectionSchemaBuilder::new(name, "nova-load");
        builder.add_field(FieldSchema::new_primary_varchar(
            PK_FIELD,
            "point id",
            false,
            self.id_max_length,
        ));
        for (vname, dim) in &dense {
            builder.add_field(FieldSchema::new_float_vector(vname.as_str(), "", *dim));
        }
        let milvus_schema = builder.build().map_err(to_other)?;

        // create_collection returns the live Collection handle.
        let collection = self.client.create_collection(milvus_schema, None).await.map_err(to_other)?;

        // Index each vector field so the collection is loadable/searchable later.
        for (vname, _) in &dense {
            let mt = metric(schema.vectors[vname.as_str()].distance.as_deref());
            let params = IndexParams::new(
                format!("{vname}_idx"),
                IndexType::IvfFlat,
                mt,
                HashMap::from([("nlist".to_string(), "128".to_string())]),
            );
            collection.create_index(vname.as_str(), params).await.map_err(to_other)?;
        }
        Ok(())
    }

    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError> {
        if points.is_empty() {
            return Ok(());
        }
        // Transpose rows → columns. Infer each vector's dim from the first point
        // (workers don't see the CollectionSchema — see the module note).
        let mut ids: Vec<String> = Vec::with_capacity(points.len());
        let mut vecs: HashMap<String, (i64, Vec<f32>)> = HashMap::new();

        for point in points {
            ids.push(match point.id {
                PointId::Integer(n) => n.to_string(),
                PointId::String(s) => s,
            });
            for (vname, value) in point.vectors {
                match value {
                    VectorValue::Dense(d) => {
                        let entry =
                            vecs.entry(vname).or_insert_with(|| (d.len() as i64, Vec::new()));
                        entry.1.extend(d);
                    }
                    VectorValue::Sparse { .. } | VectorValue::Multi(_) => {
                        return Err(StoreError::Other(format!(
                            "milvus backend supports dense vectors only; vector `{vname}` is not dense"
                        )));
                    }
                }
            }
        }

        let pk = FieldSchema::new_primary_varchar(PK_FIELD, "", false, self.id_max_length);
        let mut columns = vec![FieldColumn::new(&pk, ids)];
        for (vname, (dim, flat)) in vecs {
            let fs = FieldSchema::new_float_vector(&vname, "", dim);
            columns.push(FieldColumn::new(&fs, flat));
        }

        let collection =
            self.client.get_collection(self.collection_name.as_str()).await.map_err(to_other)?;
        collection.insert(columns, None).await.map_err(to_other)?;
        Ok(())
    }

    async fn enable_indexing(&self) -> Result<(), StoreError> {
        // Persist buffered inserts, then load the collection so it's queryable.
        let collection =
            self.client.get_collection(self.collection_name.as_str()).await.map_err(to_other)?;
        collection.flush().await.map_err(to_other)?;
        collection.load(1).await.map_err(to_other)?;
        Ok(())
    }

    async fn close(&self) -> Result<(), StoreError> {
        Ok(())
    }

    async fn delete_collection(&self) -> Result<(), StoreError> {
        let name = self.collection_name.as_str();
        if self.client.has_collection(name).await.map_err(to_other)? {
            self.client.drop_collection(name).await.map_err(to_other)?;
        }
        Ok(())
    }
}
