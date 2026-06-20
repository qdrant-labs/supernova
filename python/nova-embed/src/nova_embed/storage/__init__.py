from nova_embed.storage.base import StorageBackend

# Import the concrete backends so their @STORAGE.register decorators run.
from nova_embed.storage.s3 import S3Backend
from nova_embed.storage.huggingface import HuggingFaceBackend
from nova_embed.storage.local import LocalBackend

__all__ = ["StorageBackend", "S3Backend", "HuggingFaceBackend", "LocalBackend"]
