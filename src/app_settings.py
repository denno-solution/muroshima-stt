import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class AppSettings:
    """アプリケーション設定を管理するクラス"""
    
    def __init__(self, settings_file: str = ".app_settings.json"):
        self.settings_path = Path(__file__).parent.parent / settings_file
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        if self.settings_path.exists():
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"設定ファイルの読み込みエラー: {e}")
                return {}
        return {}
    
    def _save_settings(self):
        """設定をファイルに保存"""
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
            logger.debug(f"設定を保存しました: {self.settings_path}")
        except Exception as e:
            logger.error(f"設定ファイルの保存エラー: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """設定値を保存"""
        self.settings[key] = value
        self._save_settings()
    
    def get_selected_stt_model(self) -> Optional[str]:
        """選択されたSTTモデルを取得"""
        return self.get("selected_stt_model")
    
    def set_selected_stt_model(self, model_name: str):
        """選択されたSTTモデルを保存"""
        self.set("selected_stt_model", model_name)
    
    def get_use_structuring(self) -> bool:
        """構造化機能の有効/無効を取得"""
        return self.get("use_structuring", True)
    
    def set_use_structuring(self, use: bool):
        """構造化機能の有効/無効を保存"""
        self.set("use_structuring", use)
    
    def get_debug_mode(self) -> bool:
        """デバッグモードの有効/無効を取得"""
        return self.get("debug_mode", False)
    
    def set_debug_mode(self, debug: bool):
        """デバッグモードの有効/無効を保存"""
        self.set("debug_mode", debug)