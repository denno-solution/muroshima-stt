import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioStorage:
    """
    音声ファイルのローカルストレージ管理クラス
    低コストでサーバー側に音声ファイルを永続化保存
    """
    
    def __init__(self, storage_dir: str = "stored_audio"):
        """
        Args:
            storage_dir: 音声ファイルを保存するディレクトリ名
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"AudioStorage initialized with directory: {self.storage_dir}")
    
    def save_audio_file(self, temp_file_path: str, original_filename: str) -> Tuple[str, int]:
        """
        一時ファイルを永続化ストレージに保存
        
        Args:
            temp_file_path: 一時ファイルのパス
            original_filename: 元のファイル名
            
        Returns:
            Tuple[str, int]: (保存されたファイルパス, ファイルサイズ)
        """
        try:
            # ファイルサイズを取得
            file_size = os.path.getsize(temp_file_path)
            
            # タイムスタンプ付きのユニークなファイル名を生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ミリ秒まで
            file_ext = Path(original_filename).suffix
            safe_original = self._sanitize_filename(Path(original_filename).stem)
            stored_filename = f"{timestamp}_{safe_original}{file_ext}"
            
            # 保存先パス
            stored_path = self.storage_dir / stored_filename
            
            # ファイルをコピー
            shutil.copy2(temp_file_path, stored_path)
            
            logger.info(f"Audio file saved: {stored_path} ({file_size} bytes)")
            return str(stored_path), file_size
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {str(e)}")
            raise
    
    def get_audio_file_path(self, stored_path: str) -> Optional[str]:
        """
        保存されたファイルのパスを取得
        
        Args:
            stored_path: 保存時に返されたファイルパス
            
        Returns:
            str: ファイルが存在する場合はパス、存在しない場合はNone
        """
        file_path = Path(stored_path)
        if file_path.exists():
            return str(file_path)
        return None
    
    def delete_audio_file(self, stored_path: str) -> bool:
        """
        保存された音声ファイルを削除
        
        Args:
            stored_path: 削除対象のファイルパス
            
        Returns:
            bool: 削除成功時True、失敗時False
        """
        try:
            file_path = Path(stored_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Audio file deleted: {stored_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {stored_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete audio file {stored_path}: {str(e)}")
            return False
    
    def get_storage_stats(self) -> dict:
        """
        ストレージの使用状況を取得
        
        Returns:
            dict: ストレージ統計情報
        """
        try:
            total_files = 0
            total_size = 0
            
            if self.storage_dir.exists():
                for file_path in self.storage_dir.iterdir():
                    if file_path.is_file():
                        total_files += 1
                        total_size += file_path.stat().st_size
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "storage_directory": str(self.storage_dir)
            }
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {"error": str(e)}
    
    def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        指定日数より古いファイルを削除
        
        Args:
            days_old: 削除対象となる日数（デフォルト30日）
            
        Returns:
            int: 削除されたファイル数
        """
        try:
            if not self.storage_dir.exists():
                return 0
            
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            for file_path in self.storage_dir.iterdir():
                if file_path.is_file():
                    # ファイルの作成時刻をチェック
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            logger.info(f"Deleted old audio file: {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete old file {file_path}: {str(e)}")
            
            logger.info(f"Cleanup completed: {deleted_count} files deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {str(e)}")
            return 0
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        ファイル名を安全な文字列に変換
        
        Args:
            filename: 元のファイル名
            
        Returns:
            str: サニタイズされたファイル名
        """
        # 危険な文字を除去
        import re
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        # 長すぎる場合は切り詰め
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        return safe_name


# グローバルインスタンス
_audio_storage = None

def get_audio_storage() -> AudioStorage:
    """
    AudioStorageのシングルトンインスタンスを取得
    
    Returns:
        AudioStorage: AudioStorageのインスタンス
    """
    global _audio_storage
    if _audio_storage is None:
        _audio_storage = AudioStorage()
    return _audio_storage