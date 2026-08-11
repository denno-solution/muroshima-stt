import os
import sys
from dataclasses import dataclass
from pathlib import Path
import importlib.util
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# スクリプトディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent / "scripts"))


@dataclass
class STTResult:
    """文字起こし結果(単語タイムスタンプ付き)。

    words は {text, start, end, type, speaker_id?} の配列(秒、STT入力音声基準)。
    単語タイムスタンプ非対応のモデルでは None。
    """

    text: Optional[str]
    words: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


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

    def transcribe_detailed(self, audio_file_path: str) -> STTResult:
        """文字起こし+単語タイムスタンプ取得(対応モデルのみ)。

        `transcribe_audio_file_detailed` を持つモジュール(現状 ElevenLabs)は
        単語タイムスタンプ付きで返す。それ以外は従来の transcribe() の結果を
        words=None で包む(後方互換)。
        """
        if hasattr(self.module, "transcribe_audio_file_detailed"):
            detailed = self.module.transcribe_audio_file_detailed(audio_file_path)
            if isinstance(detailed, dict):
                return STTResult(
                    text=detailed.get("text"),
                    words=detailed.get("words") or None,
                    error=detailed.get("error"),
                )

        result = self.transcribe(audio_file_path)
        if isinstance(result, tuple) and result[0] is None:
            error = result[1] if len(result) > 1 else "STT failed"
            return STTResult(text=None, words=None, error=error)
        return STTResult(text=result, words=None, error=None)
    
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