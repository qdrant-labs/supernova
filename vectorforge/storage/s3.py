import aiobotocore.session


async def upload_to_s3(local_path: str, bucket: str, prefix: str):
    filename = local_path.split("/")[-1]
    key = f"{prefix}/{filename}"

    session = aiobotocore.session.get_session()
    async with session.create_client("s3") as client:
        with open(local_path, "rb") as f:
            await client.put_object(Bucket=bucket, Key=key, Body=f)

    print(f"Uploaded s3://{bucket}/{key}")
