import { S3Client, ListObjectsV2Command, GetObjectCommand } from "@aws-sdk/client-s3";
import { writeFileSync, mkdirSync } from "fs";
import { join } from "path";

const BUCKET = "qdrant--vectorforge";
const OUTPUT_DIR = join(__dirname, "..", "data");
const OUTPUT_FILE = join(OUTPUT_DIR, "manifests.json");

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

async function fetchManifests(): Promise<void> {
  const client = new S3Client({});

  console.log(`Listing objects in s3://${BUCKET} looking for _manifest.json files...`);

  const manifests: Manifest[] = [];
  let continuationToken: string | undefined;

  do {
    const listCommand = new ListObjectsV2Command({
      Bucket: BUCKET,
      ContinuationToken: continuationToken,
    });

    const listResponse = await client.send(listCommand);

    const manifestKeys = (listResponse.Contents ?? [])
      .map((obj) => obj.Key!)
      .filter((key) => key.endsWith("_manifest.json"));

    for (const key of manifestKeys) {
      console.log(`  Fetching s3://${BUCKET}/${key}`);
      const getCommand = new GetObjectCommand({ Bucket: BUCKET, Key: key });
      const getResponse = await client.send(getCommand);
      const body = await getResponse.Body!.transformToString();
      const manifest: Manifest = JSON.parse(body);
      manifests.push(manifest);
    }

    continuationToken = listResponse.NextContinuationToken;
  } while (continuationToken);

  // Sort by created_at descending
  manifests.sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );

  mkdirSync(OUTPUT_DIR, { recursive: true });
  writeFileSync(OUTPUT_FILE, JSON.stringify(manifests, null, 2) + "\n");

  console.log(`\nWrote ${manifests.length} manifests to ${OUTPUT_FILE}`);
}

fetchManifests().catch((err) => {
  console.error("Failed to fetch manifests:", err);
  process.exit(1);
});
