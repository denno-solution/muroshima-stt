#!/usr/bin/env python3
"""
VAD機能の基本テスト
"""

import sys
import logging
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vad_processor import VADProcessor

def test_vad_initialization():
    """VAD初期化テスト"""
    try:
        processor = VADProcessor(aggressiveness=2)
        print("✅ VADProcessor初期化成功")
        return True
    except Exception as e:
        print(f"❌ VADProcessor初期化失敗: {e}")
        return False

def test_ffmpeg():
    """FFmpeg利用可能性テスト"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg利用可能")
            return True
        else:
            print("❌ FFmpeg実行エラー")
            return False
    except FileNotFoundError:
        print("❌ FFmpegが見つかりません")
        return False
    except Exception as e:
        print(f"❌ FFmpegテストエラー: {e}")
        return False

def test_webrtcvad():
    """webrtcvad動作テスト"""
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)
        
        # ダミーの16bit PCMデータを作成（16kHz、30ms）
        import struct
        frame_size = int(16000 * 0.03) * 2  # 30ms, 16bit
        dummy_frame = struct.pack('<' + 'h' * (frame_size // 2), *([0] * (frame_size // 2)))
        
        # VAD実行
        is_speech = vad.is_speech(dummy_frame, 16000)
        print(f"✅ webrtcvad動作確認完了（無音検出: {not is_speech}）")
        return True
    except Exception as e:
        print(f"❌ webrtcvadエラー: {e}")
        return False

def main():
    """テストメイン関数"""
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 VAD機能テスト開始")
    print("=" * 50)
    
    tests = [
        ("webrtcvad動作テスト", test_webrtcvad),
        ("FFmpeg利用可能性テスト", test_ffmpeg),
        ("VAD初期化テスト", test_vad_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        result = test_func()
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
        print("\n🎉 すべてのテストに合格しました！VAD機能の準備完了です。")
        return 0
    else:
        print("\n⚠️  いくつかのテストが失敗しました。環境設定を確認してください。")
        return 1

if __name__ == "__main__":
    sys.exit(main())