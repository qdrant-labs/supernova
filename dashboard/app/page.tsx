import manifests from "@/data/manifests.json";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Manifest {
  source: string;
  embedder: string;
  dimensions: number;
  chunk_size: number;
  max_tokens: number;
  num_workers: number;
  flush_threshold: number;
  total_records: number;
  total_batches: number;
  elapsed_seconds: number;
  records_per_second: number;
  created_at: string;
  s3_bucket: string;
  s3_prefix: string;
}

function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function Home() {
  const sorted = [...(manifests as Manifest[])].sort(
    (a, b) =>
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );

  const totalDatasets = sorted.length;
  const totalRecords = sorted.reduce((sum, m) => sum + m.total_records, 0);

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-border/40 px-6 py-8">
        <div className="mx-auto max-w-7xl">
          <h1 className="text-3xl font-bold tracking-tight font-mono">
            vectorforge
          </h1>
          <p className="mt-1 text-muted-foreground">
            Embedding datasets at scale
          </p>
        </div>
      </header>

      {/* Summary bar */}
      <div className="border-b border-border/40 bg-muted/30 px-6 py-4">
        <div className="mx-auto flex max-w-7xl gap-8">
          <div>
            <span className="text-sm text-muted-foreground">
              Total datasets
            </span>
            <p className="text-2xl font-semibold tabular-nums">
              {totalDatasets}
            </p>
          </div>
          <div>
            <span className="text-sm text-muted-foreground">
              Total records
            </span>
            <p className="text-2xl font-semibold tabular-nums">
              {formatNumber(totalRecords)}
            </p>
          </div>
        </div>
      </div>

      {/* Card grid */}
      <main className="mx-auto max-w-7xl px-6 py-8">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {sorted.map((m) => (
            <Card key={m.s3_prefix}>
              <CardHeader>
                <CardTitle className="text-base font-semibold leading-tight">
                  {m.source}
                </CardTitle>
                <CardDescription>
                  <Badge variant="secondary" className="mt-1">
                    {m.embedder}
                  </Badge>
                </CardDescription>
              </CardHeader>
              <CardContent>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                  <div>
                    <dt className="text-muted-foreground">Vectors</dt>
                    <dd className="font-medium tabular-nums">
                      {formatNumber(m.total_records)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Throughput</dt>
                    <dd className="font-medium tabular-nums">
                      {formatNumber(Math.round(m.records_per_second))}/s
                    </dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Dimensions</dt>
                    <dd className="font-medium tabular-nums">
                      {formatNumber(m.dimensions)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Created</dt>
                    <dd className="font-medium">
                      {formatDate(m.created_at)}
                    </dd>
                  </div>
                  <div className="col-span-2">
                    <dt className="text-muted-foreground">S3 path</dt>
                    <dd className="font-mono text-xs text-muted-foreground/80 truncate">
                      s3://{m.s3_bucket}/{m.s3_prefix}
                    </dd>
                  </div>
                </dl>
              </CardContent>
            </Card>
          ))}
        </div>
      </main>
    </div>
  );
}
