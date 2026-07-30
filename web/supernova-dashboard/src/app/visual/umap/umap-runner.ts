import { UMAP } from 'umap-js';

export interface UmapRunnerOptions {
  nNeighbors?: number;
  minDist?: number;
  nComponents?: number;
  onProgress?: (epoch: number, totalEpochs: number) => void;
}

export interface UmapCoordinate {
  x: number;
  y: number;
}

export async function runUmap(
  vectors: number[][],
  options: UmapRunnerOptions = {},
): Promise<UmapCoordinate[]> {
  if (vectors.length === 0) {
    return [];
  }

  const nNeighbors = Math.min(options.nNeighbors ?? 15, Math.max(2, vectors.length - 1));
  const nComponents = options.nComponents ?? 2;
  const umap = new UMAP({
    nComponents,
    nNeighbors,
    minDist: options.minDist ?? 0.1,
    distanceFn: cosineDistance,
  });

  const totalEpochs = umap.initializeFit(vectors);
  const embedding = await umap.fitAsync(vectors, (epoch) => {
    options.onProgress?.(epoch, totalEpochs);
  });

  return embedding.map((coords) => ({
    x: coords[0] ?? 0,
    y: coords[1] ?? 0,
  }));
}

function cosineDistance(x: number[], y: number[]): number {
  let result = 0;
  let normX = 0;
  let normY = 0;

  for (let i = 0; i < x.length; i++) {
    result += x[i] * y[i];
    normX += x[i] ** 2;
    normY += y[i] ** 2;
  }

  if (normX === 0 && normY === 0) {
    return 0;
  }

  if (normX === 0 || normY === 0) {
    return 1;
  }

  return 1 - result / Math.sqrt(normX * normY);
}
