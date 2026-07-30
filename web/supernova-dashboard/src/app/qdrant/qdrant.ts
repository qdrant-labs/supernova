import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

export interface UmapPoint {
  id: string | number;
  vector: number[];
  payload?: Record<string, unknown>;
}

export interface EmbeddingSearchPoint {
  id: string | number;
  score: number;
  vector: number[];
  payload?: Record<string, unknown>;
}

export interface RandomDenseQueryResult {
  vectorName?: string;
  queryVector: number[];
  points: EmbeddingSearchPoint[];
}

@Injectable({
  providedIn: 'root',
})
export class Qdrant {
  private readonly http = inject(HttpClient);

  async getCollections(): Promise<string[]> {
    const response = await firstValueFrom(
      this.http.get<{ collections: string[] }>('/api/v1/qdrant/collections'),
    );
    return response.collections;
  }

  async randomDenseQuery(collectionName: string, limit = 50): Promise<RandomDenseQueryResult> {
    const response = await firstValueFrom(
      this.http.post<{ query_vector: number[]; points: EmbeddingSearchPoint[] }>(
        '/api/v1/qdrant/random-query',
        {
          collection_name: collectionName,
          limit,
        },
      ),
    );

    return {
      queryVector: response.query_vector,
      points: response.points,
    };
  }

  async scrollVectors(collectionName: string, limit = 2000): Promise<UmapPoint[]> {
    const response = await firstValueFrom(
      this.http.post<{ points: UmapPoint[] }>('/api/v1/qdrant/scroll', {
        collection_name: collectionName,
        limit,
      }),
    );
    return response.points;
  }
}
