#!/usr/bin/env python3
"""
VAD機能の簡易テスト（FFmpegなし）
"""

import sys
import struct
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_webrtcvad_basic():
    """webrtcvadの基本的な動作テスト"""
    try:
        import webrtcvad
        print("✅ webrtcvadインポート成功")
        
        # VAD初期化
        vad = webrtcvad.Vad(2)
        print("✅ VAD初期化成功")
        
        # ダミーの16bit PCMデータを作成（16kHz、30ms）
        sample_rate = 16000
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16bit = 2bytes
        
        # 無音フレーム
        silence_frame = struct.pack('<' + 'h' * (frame_size // 2), *([0] * (frame_size // 2)))
        is_speech_silence = vad.is_speech(silence_frame, sample_rate)
        print(f"✅ 無音フレーム判定: {is_speech_silence} (Falseが期待値)")
        
        # ノイズフレーム（ランダムな値）
        import random
        random.seed(42)  # 再現可能な結果のため
        noise_values = [random.randint(-1000, 1000) for _ in range(frame_size // 2)]
        noise_frame = struct.pack('<' + 'h' * len(noise_values), *noise_values)
        is_speech_noise = vad.is_speech(noise_frame, sample_rate)
        print(f"✅ ノイズフレーム判定: {is_speech_noise} (結果は可変)")
        
        return True
        
    except Exception as e:
        print(f"❌ webrtcvadテストエラー: {e}")
        return False

def test_vad_processor_init():
    """VADProcessorの初期化テスト"""
    try:
        from vad_processor import VADProcessor
        
        processor = VADProcessor(aggressiveness=2)
        print("✅ VADProcessor初期化成功")
        
        # VAD設定確認
        print(f"✅ VAD厳しさ設定: {processor.aggressiveness}")
        
        return True
        
    except Exception as e:
        print(f"❌ VADProcessorテストエラー: {e}")
        return False

def test_vad_elevenlabs_init():
    """VADElevenLabsSTTの初期化テスト（APIキーなし）"""
    try:
        # 環境変数を一時的に設定
        import os
        os.environ['ELEVENLABS_API_KEY'] = 'test-key-for-init-only'
        
        from vad_elevenlabs import VADElevenLabsSTT
        
        stt = VADElevenLabsSTT(api_key='test-key', vad_aggressiveness=2)
        print("✅ VADElevenLabsSTT初期化成功")
        
        # 設定確認
        print(f"✅ VADプロセッサー利用可能: {stt.vad_processor is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ VADElevenLabsSTTテストエラー: {e}")
        return False

def main():
    """テストメイン関数"""
    print("🔧 VAD機能簡易テスト開始")
    print("=" * 50)
    
    tests = [
        ("webrtcvad基本動作テスト", test_webrtcvad_basic),
        ("VADProcessor初期化テスト", test_vad_processor_init),
        ("VADElevenLabsSTT初期化テスト", test_vad_elevenlabs_init),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            result = test_func()
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
            result = False
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("🏁 テスト結果")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 結果: {passed}/{len(tests)} テスト通過")
    
    if passed == len(tests):
        print("\n🎉 基本的なVAD機能のテストに合格しました！")
        print("💡 実際の音声処理にはFFmpegが必要です。")
        return 0
    else:
        print("\n⚠️  いくつかのテストが失敗しました。")
        return 1

if __name__ == "__main__":
    sys.exit(main())