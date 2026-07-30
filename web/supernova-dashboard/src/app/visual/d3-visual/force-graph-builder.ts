import * as d3 from 'd3';

import { EmbeddingSearchPoint } from '../../qdrant/qdrant';

export interface ForceGraphNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  score?: number;
  isQuery?: boolean;
}

export interface ForceGraphLink extends d3.SimulationLinkDatum<ForceGraphNode> {
  similarity: number;
}

export interface ForceGraphOptions {
  minSimilarity?: number;
  maxNeighbors?: number;
  queryVector?: number[];
}

const QUERY_NODE_ID = '__query__';

export function buildEmbeddingGraph(
  points: EmbeddingSearchPoint[],
  options: ForceGraphOptions = {},
): { nodes: ForceGraphNode[]; links: ForceGraphLink[] } {
  const minSimilarity = options.minSimilarity ?? 0.55;
  const maxNeighbors = options.maxNeighbors ?? 4;

  const nodes: ForceGraphNode[] = points.map((point) => ({
    id: String(point.id),
    label: String(point.id),
    score: point.score,
    isQuery: false,
  }));

  const links: ForceGraphLink[] = [];
  const linkKeys = new Set<string>();

  if (options.queryVector) {
    nodes.unshift({
      id: QUERY_NODE_ID,
      label: 'query',
      isQuery: true,
    });

    for (const point of points) {
      const similarity =
        point.score ?? cosineSimilarity(options.queryVector, point.vector);
      links.push({
        source: QUERY_NODE_ID,
        target: String(point.id),
        similarity,
      });
    }
  }

  for (let i = 0; i < points.length; i++) {
    const neighbors = points
      .map((other, index) => ({
        index,
        similarity: cosineSimilarity(points[i].vector, other.vector),
      }))
      .filter((entry) => entry.index !== i)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, maxNeighbors)
      .filter((entry) => entry.similarity >= minSimilarity);

    for (const neighbor of neighbors) {
      const sourceId = String(points[i].id);
      const targetId = String(points[neighbor.index].id);
      const key =
        sourceId < targetId ? `${sourceId}|${targetId}` : `${targetId}|${sourceId}`;

      if (linkKeys.has(key)) {
        continue;
      }

      linkKeys.add(key);
      links.push({
        source: sourceId,
        target: targetId,
        similarity: neighbor.similarity,
      });
    }
  }

  return { nodes, links };
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
