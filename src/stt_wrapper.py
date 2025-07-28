import os
import sys
from pathlib import Path
import importlib.util
from typing import Dict, Any, Optional
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
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.module_name = self.AVAILABLE_MODELS.get(model_name)
        if not self.module_name:
            raise ValueError(f"Unknown model: {model_name}")
        
        # モジュールを動的にインポート
        try:
            self.module = importlib.import_module(self.module_name)
        except ImportError as e:
            raise ImportError(f"Failed to import {self.module_name}: {e}")
    
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """音声ファイルを文字起こしする"""
        if hasattr(self.module, 'transcribe_audio_file'):
            result = self.module.transcribe_audio_file(audio_file_path)
            # ElevenLabsなど一部のモジュールはエラー時にタプルを返す
            if isinstance(result, tuple) and result[0] is None:
                # エラーメッセージを含むタプルをそのまま返す
                return result
            return result
        else:
            raise AttributeError(f"{self.module_name} does not have transcribe_audio_file function")
    
    @classmethod
    def get_available_models(cls) -> list:
        """利用可能なモデル名のリストを返す"""
        return list(cls.AVAILABLE_MODELS.keys())
    
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