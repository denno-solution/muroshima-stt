import os
import logging
from pathlib import Path
from datetime import datetime
import uuid

try:
    from supabase import create_client  # type: ignore
except Exception:  # ランタイムで未インストールでも他機能は動かす
    create_client = None  # type: ignore


logger = logging.getLogger(__name__)


def _get_supabase_client():
    if create_client is None:
        return None
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Supabaseクライアント初期化に失敗: {e}")
        return None


def upload_audio_to_supabase(local_path: str, *, content_type: str) -> dict | None:
    """音声ファイルをSupabase Storageにアップロード。
    成功時は {bucket, path, public_url} を返す。失敗時はNone。
    """
    sb = _get_supabase_client()
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "stt-audio")
    if not sb:
        return None
    now = datetime.utcnow()
    ext = Path(local_path).suffix.lower() or ".wav"
    unique = uuid.uuid4().hex
    dest_path = f"recordings/{now:%Y/%m/%d}/{unique}{ext}"
    try:
        with open(local_path, "rb") as f:
            sb.storage.from_(bucket).upload(
                dest_path,
                f,
                {"content-type": content_type, "x-upsert": "true"},
            )
        public_url = None
        try:
            public_url = sb.storage.from_(bucket).get_public_url(dest_path)
        except Exception:
            public_url = None
        return {"bucket": bucket, "path": dest_path, "public_url": public_url}
    except Exception as e:
        logger.error(f"Supabase Storage アップロード失敗: {e}")
        return None

