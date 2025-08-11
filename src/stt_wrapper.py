import os
import sys
from pathlib import Path
import importlib.util
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# スクリプトディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

class STTModelWrapper:
    """各STTモデルスクリプトを統一インターフェースで扱うラッパークラス"""
    
    AVAILABLE_MODELS = {
        "OpenAI": "transcribe_openai",
        "Google Cloud (Chirp)": "transcribe_google",
        "Amazon Transcribe": "transcribe_amazon",
        "Azure Speech": "transcribe_azure",
        "ElevenLabs": "transcribe_elevenlabs"
    }
    
    # VAD対応モデル
    VAD_SUPPORTED_MODELS = {"ElevenLabs"}
    
    def __init__(self, model_name: str, enable_vad: bool = False):
        self.model_name = model_name
        self.enable_vad = enable_vad
        self.module_name = self.AVAILABLE_MODELS.get(model_name)
        if not self.module_name:
            raise ValueError(f"Unknown model: {model_name}")
        
        # モジュールを動的にインポート
        try:
            self.module = importlib.import_module(self.module_name)
        except ImportError as e:
            raise ImportError(f"Failed to import {self.module_name}: {e}")
        
        # VAD対応モジュールの初期化
        self.vad_module = None
        if enable_vad and model_name in self.VAD_SUPPORTED_MODELS:
            try:
                from vad_elevenlabs import VADElevenLabsSTT
                self.vad_module = VADElevenLabsSTT()
            except ImportError as e:
                raise ImportError(f"Failed to import VAD module: {e}")
    
    def transcribe(self, audio_file_path: str, vad_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """音声ファイルを文字起こしする"""
        # VAD対応モジュールが利用可能で、VADが有効化されている場合
        if self.vad_module is not None and self.enable_vad:
            vad_params = vad_params or {}
            transcript, metadata = self.vad_module.transcribe_file_with_vad(
                audio_file_path, **vad_params
            )
            if transcript is None:
                # エラーの場合はタプルで返す（既存の互換性のため）
                return (None, metadata.get('error', 'VAD processing failed'))
            return transcript
        
        # 従来のSTT処理
        if hasattr(self.module, 'transcribe_audio_file'):
            result = self.module.transcribe_audio_file(audio_file_path)
            # ElevenLabsなど一部のモジュールはエラー時にタプルを返す
            if isinstance(result, tuple) and result[0] is None:
                # エラーメッセージを含むタプルをそのまま返す
                return result
            return result
        else:
            raise AttributeError(f"{self.module_name} does not have transcribe_audio_file function")
    
    def transcribe_with_metadata(self, audio_file_path: str, vad_params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """音声ファイルを文字起こしし、メタデータも返す"""
        # VAD対応モジュールが利用可能で、VADが有効化されている場合
        if self.vad_module is not None and self.enable_vad:
            vad_params = vad_params or {}
            return self.vad_module.transcribe_file_with_vad(
                audio_file_path, **vad_params
            )
        
        # 従来のSTT処理（メタデータなし）
        result = self.transcribe(audio_file_path)
        if isinstance(result, tuple):
            # エラーケース
            return result[0], {"error": result[1]}
        else:
            return result, {}
    
    @classmethod
    def get_available_models(cls) -> list:
        """利用可能なモデル名のリストを返す"""
        return list(cls.AVAILABLE_MODELS.keys())
    
    @classmethod
    def get_vad_supported_models(cls) -> list:
        """VAD対応モデル名のリストを返す"""
        return list(cls.VAD_SUPPORTED_MODELS)
    
    def is_vad_supported(self) -> bool:
        """現在のモデルがVADに対応しているかチェック"""
        return self.model_name in self.VAD_SUPPORTED_MODELS
    
    def is_vad_enabled(self) -> bool:
        """VAD機能が有効化されているかチェック"""
        return self.enable_vad and self.vad_module is not None
    
    def check_requirements(self) -> Dict[str, bool]:
        """必要な環境変数やAPIキーの設定状況をチェック"""
        requirements = {}
        
        if self.model_name == "OpenAI":
            requirements["OPENAI_API_KEY"] = bool(os.getenv("OPENAI_API_KEY"))
        elif self.model_name == "Google Cloud (Chirp)":
            requirements["GOOGLE_CLOUD_PROJECT"] = bool(os.getenv("GOOGLE_CLOUD_PROJECT"))
            requirements["GOOGLE_APPLICATION_CREDENTIALS"] = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        elif self.model_name == "Amazon Transcribe":
            requirements["AWS_ACCESS_KEY_ID"] = bool(os.getenv("AWS_ACCESS_KEY_ID"))
            requirements["AWS_SECRET_ACCESS_KEY"] = bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
        elif self.model_name == "Azure Speech":
            requirements["AZURE_SPEECH_KEY"] = bool(os.getenv("AZURE_SPEECH_KEY"))
            requirements["AZURE_SPEECH_REGION"] = bool(os.getenv("AZURE_SPEECH_REGION"))
        elif self.model_name == "ElevenLabs":
            requirements["ELEVENLABS_API_KEY"] = bool(os.getenv("ELEVENLABS_API_KEY"))
        
        return requirements