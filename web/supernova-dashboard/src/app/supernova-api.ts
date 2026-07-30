import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface JobAccepted {
  job_id: string;
}

export interface JobRecord {
  id: string;
  kind: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  created_at_ms: number;
  started_at_ms?: number;
  finished_at_ms?: number;
  result?: unknown;
  error?: string;
  logs: string[];
}

export interface QdrantCollectionsResponse {
  collections: string[];
}

export interface DistLoadRequest {
  config_path?: string;
  config_yaml?: string;
  resources?: string;
  num_jobs?: number;
  pool_name?: string;
  dry_run?: boolean;
  finalize?: boolean;
  catalog_path?: string;
  catalog_remote_dir?: string;
  build_catalog_input?: string;
  build_catalog_output?: string;
  build_catalog_resume?: boolean;
}

export interface DistStormRequest {
  config_path?: string;
  config_yaml?: string;
  resources?: string;
  num_jobs: number;
  pool_name?: string;
  dry_run?: boolean;
  stage_query_source?: string;
  query_source_remote_dir?: string;
}

@Injectable({ providedIn: 'root' })
export class SupernovaApi {
  private readonly http = inject(HttpClient);

  getCollections(): Observable<QdrantCollectionsResponse> {
    return this.http.get<QdrantCollectionsResponse>('/api/v1/qdrant/collections');
  }

  submitLoadRun(configYaml: string): Observable<JobAccepted> {
    return this.http.post<JobAccepted>('/api/v1/load/run', { config_yaml: configYaml });
  }

  submitStormRun(configYaml: string): Observable<JobAccepted> {
    return this.http.post<JobAccepted>('/api/v1/storm/run', { config_yaml: configYaml });
  }

  submitDistLoad(req: DistLoadRequest): Observable<JobAccepted> {
    return this.http.post<JobAccepted>('/api/v1/dist/load', req);
  }

  submitDistStorm(req: DistStormRequest): Observable<JobAccepted> {
    return this.http.post<JobAccepted>('/api/v1/dist/storm', req);
  }

  listJobs(): Observable<JobRecord[]> {
    return this.http.get<JobRecord[]>('/api/v1/jobs');
  }

  getJob(jobId: string): Observable<JobRecord> {
    return this.http.get<JobRecord>(`/api/v1/jobs/${jobId}`);
  }

  cancelJob(jobId: string): Observable<{ job_id: string; cancelled: boolean }> {
    return this.http.post<{ job_id: string; cancelled: boolean }>(
      `/api/v1/jobs/${jobId}/cancel`,
      {}
    );
  }
}
