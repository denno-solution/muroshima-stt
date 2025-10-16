from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.client import Config


@dataclass
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    prefix: str = ""
    public_base_url: Optional[str] = None  # e.g. https://pub-xxxxxx.r2.dev/my-bucket


def load_r2_config_from_env() -> Optional[R2Config]:
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")
    prefix = os.getenv("R2_PREFIX", "")
    public_base_url = os.getenv("R2_PUBLIC_BASE_URL")

    if not all([account_id, access_key_id, secret_access_key, bucket_name]):
        return None

    if prefix and not prefix.endswith("/"):
        prefix += "/"

    return R2Config(
        account_id=account_id,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        bucket_name=bucket_name,
        prefix=prefix,
        public_base_url=public_base_url,
    )


def _build_s3_client(cfg: R2Config):
    endpoint = f"https://{cfg.account_id}.r2.cloudflarestorage.com"
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    return s3


def guess_content_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".wav"):
        return "audio/wav"
    if lower.endswith(".mp3"):
        return "audio/mpeg"
    if lower.endswith(".m4a"):
        return "audio/mp4"
    if lower.endswith(".flac"):
        return "audio/flac"
    if lower.endswith(".ogg"):
        return "audio/ogg"
    if lower.endswith(".webm"):
        return "audio/webm"
    return "application/octet-stream"


def upload_file_to_r2(local_path: str, key: str, cfg: Optional[R2Config] = None) -> dict:
    """Upload a local file to Cloudflare R2.

    Returns a dict {"bucket": str, "key": str, "url": Optional[str]}.
    """
    if cfg is None:
        cfg = load_r2_config_from_env()
    if cfg is None:
        raise RuntimeError("R2 configuration is missing. Set R2_* env vars.")

    s3 = _build_s3_client(cfg)

    if cfg.prefix:
        key = f"{cfg.prefix}{key}" if not key.startswith(cfg.prefix) else key

    content_type = guess_content_type(local_path)

    s3.upload_file(
        local_path,
        cfg.bucket_name,
        key,
        ExtraArgs={"ContentType": content_type},
    )

    url = None
    if cfg.public_base_url:
        base = cfg.public_base_url.rstrip("/")
        url = f"{base}/{key}"

    return {"bucket": cfg.bucket_name, "key": key, "url": url}


def build_object_key_for_filename(filename: str, cfg: Optional[R2Config] = None) -> Optional[str]:
    """Return the R2 object key for a given local filename considering prefix.

    - If R2 is not configured, returns None.
    - If `R2_PREFIX` is set, prepend it (ensuring a single slash).
    - If filename already starts with the prefix, use as-is.
    """
    if cfg is None:
        cfg = load_r2_config_from_env()
    if cfg is None:
        return None

    key = filename.lstrip("/")
    if cfg.prefix:
        if not key.startswith(cfg.prefix):
            key = f"{cfg.prefix}{key}"
    return key


def build_public_url_for_key(key: str, cfg: Optional[R2Config] = None) -> Optional[str]:
    """Build a public URL using `R2_PUBLIC_BASE_URL` if available.

    Example base: https://pub-xxxxxx.r2.dev/my-bucket
    Result:       {base}/{key}
    """
    if cfg is None:
        cfg = load_r2_config_from_env()
    if cfg is None or not cfg.public_base_url:
        return None
    base = cfg.public_base_url.rstrip("/")
    key = key.lstrip("/")
    return f"{base}/{key}"


def generate_presigned_get_url(
    key: str,
    expires_in: int = 900,
    cfg: Optional[R2Config] = None,
) -> Optional[str]:
    """Generate a time-limited signed URL for GET on a private R2 object.

    Returns None if R2 is not configured or on failure.
    """
    if cfg is None:
        cfg = load_r2_config_from_env()
    if cfg is None:
        return None

    try:
        s3 = _build_s3_client(cfg)
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": cfg.bucket_name, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except Exception:
        return None
