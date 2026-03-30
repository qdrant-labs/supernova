import logging

import aiobotocore.session
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


async def ensure_bucket_exists(client, bucket: str):
    try:
        await client.head_bucket(Bucket=bucket)
    except ClientError:
        logger.info("Bucket %s does not exist, creating it", bucket)
        await client.create_bucket(Bucket=bucket)


async def upload_to_s3(local_path: str, bucket: str, prefix: str):
    filename = local_path.split("/")[-1]
    key = f"{prefix}/{filename}"

    session = aiobotocore.session.get_session()
    async with session.create_client("s3") as client:
        await ensure_bucket_exists(client, bucket)
        with open(local_path, "rb") as f:
            await client.put_object(Bucket=bucket, Key=key, Body=f)

    logger.info("Uploaded s3://%s/%s", bucket, key)


async def upload_bytes_to_s3(data: bytes, bucket: str, key: str):
    session = aiobotocore.session.get_session()
    async with session.create_client("s3") as client:
        await ensure_bucket_exists(client, bucket)
        await client.put_object(Bucket=bucket, Key=key, Body=data)

    logger.info("Uploaded s3://%s/%s", bucket, key)
