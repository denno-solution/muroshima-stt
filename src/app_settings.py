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
    
    def get_auto_reload_env(self) -> bool:
        """環境変数の自動リロード機能の有効/無効を取得"""
        return self.get("auto_reload_env", True)
    
    def set_auto_reload_env(self, auto_reload: bool):
        """環境変数の自動リロード機能の有効/無効を保存"""
        self.set("auto_reload_env", auto_reload)

    # --- VAD（非音声区間カット）設定 ---
    def get_use_vad(self) -> bool:
        # 互換: 旧キー `vad_enabled` を読み取り、未設定なら既定は True（ON）
        if "use_vad" in self.settings:
            return bool(self.settings["use_vad"])
        if "vad_enabled" in self.settings:
            # マイグレーション: 新キーへコピー
            try:
                self.settings["use_vad"] = bool(self.settings["vad_enabled"])
                self._save_settings()
            except Exception:
                pass
            return bool(self.settings["vad_enabled"])
        return True

    def set_use_vad(self, use: bool):
        self.set("use_vad", use)

    def get_vad_aggressiveness(self) -> int:
        # 0(緩い)〜3(厳しめ)
        return int(self.get("vad_aggressiveness", 2))

    def set_vad_aggressiveness(self, val: int):
        self.set("vad_aggressiveness", int(max(0, min(3, val))))
