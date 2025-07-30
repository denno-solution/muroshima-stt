const express = require('express');
const cors = require('cors');
const app = express();


// 🔸 CORS は最優先で設定
app.use(cors({
  origin: '*', // 必要なら 'http://localhost:8080' などに限定可能
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type']
}));

const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const tmp = require('tmp');
const { SpeechClient } = require('@google-cloud/speech');
const { Storage } = require('@google-cloud/storage');
const { v4: uuidv4 } = require('uuid');
const dayjs = require('dayjs'); // 必要なら導入: npm install dayjs


const upload = multer({ dest: 'uploads/' });
const PORT = process.env.PORT || 8080;
const BUCKET_NAME = 'speak-to-text1'; // GCSバケット名

app.use(express.static('public')); // ← これで HTML/CSS/JSを提供

const speechClient = new SpeechClient();
const storage = new Storage();

const { google } = require('googleapis');

// Secret の存在確認ログ
const keyPath = '/secrets/SHEETS-JSON-KYE';
if (fs.existsSync(keyPath)) {
  console.log('✅ Secret key file found:', keyPath);
} else {
  console.error('❌ Secret key file not found at:', keyPath);
}

async function appendToSheet({ filename,text }) {
  console.log("📁 appendToSheet に渡された filename:", filename);
  if (!filename) throw new Error("❌ filename が未定義です");

  const auth = new google.auth.GoogleAuth({
    keyFile: '/secrets/SHEETS-JSON-KYE',
    scopes: ['https://www.googleapis.com/auth/spreadsheets']
    
  });

  const client = await auth.getClient();
  auth.getClient().catch(e => {
    console.error("❌ Sheets認証エラー:", e);
  });
  
  const sheets = google.sheets({ version: 'v4', auth: client });

  const spreadsheetId = '1TGXWirQmz2Nh96hXMYIIwUZga17n5juYHQjOFGboTJE';
  const sheetName = 'SHEET4';
  const range = `${sheetName}!A1`; // appendなのでA1でOK

  const now = new Date().toISOString();

  // 🔒 署名付きURLを生成
  const [signedUrl] = await storage
    .bucket(BUCKET_NAME)
    .file(filename)
    .getSignedUrl({
      version: 'v4',
      action: 'read',
      expires: Date.now() + 60 * 60 * 1000 // 1時間有効
    });
  
  const values = [
    [now, signedUrl, text]
  ];

  await sheets.spreadsheets.values.append({
    spreadsheetId,
    range,
    valueInputOption: 'USER_ENTERED',
    insertDataOption: 'INSERT_ROWS',
    requestBody: { values }
  });

  console.log('✅ スプレッドシートに追記完了:', values);
}




app.post('/webhook', upload.single('audio'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No audio file received' });

  try {
    const inputPath = req.file.path;
    const outputPath = tmp.tmpNameSync({ postfix: '.flac' });
    const fileNameInGCS = `converted-audio/audio-${uuidv4()}.flac`;

    // ffmpegでflacへ変換
    const ffmpegCmd = `ffmpeg -y -i "${inputPath}" -threads 2 -af "highpass=f=150,lowpass=f=3800,volume=3.5" -ac 1 -ar 16000 -sample_fmt s16 "${outputPath}"`;
    console.log('🎛️ 変換コマンド:', ffmpegCmd);

    await new Promise((resolve, reject) => {
      exec(ffmpegCmd, { timeout: 60000 }, (err, stdout, stderr) => {
        if (err) {
          console.error('❌ ffmpeg error:', err);
          reject(err);
        } else {
          console.log('✅ ffmpeg完了:', outputPath);
          resolve();
        }
      });
    });

    // GCSにアップロード
    await storage.bucket(BUCKET_NAME).upload(outputPath, {
      destination: fileNameInGCS,
      
      metadata: {
        contentType: 'audio/flac'
      }
    });
    console.log(`✅ GCSアップロード: gs://${BUCKET_NAME}/${fileNameInGCS}`);
    //const publicUrl = `https://storage.googleapis.com/${BUCKET_NAME}/${fileNameInGCS}`;


    // 音声をBase64に変換してSTT
    const audioBytes = fs.readFileSync(outputPath).toString('base64');
    const sttRequest = {
      config: {
        encoding: 'FLAC',
        sampleRateHertz: 16000,
        languageCode: 'ja-JP',
        enableAutomaticPunctuation: true,
        model: 'default',
        useEnhanced: true,
      },
      audio: { content: audioBytes },
    };

    const [responseSTT] = await speechClient.recognize(sttRequest);

    let transcription = '';
    if (responseSTT.results && responseSTT.results.length > 0) {
      transcription = responseSTT.results
      .map(result => result.alternatives[0].transcript)
      .join('\n')
      .trim();
    }

    console.log('🎤 文字起こし:', transcription);

    // テキストファイルとして保存
    const transcriptFilePath = `/tmp/${path.basename(fileNameInGCS, '.flac')}.txt`;
      fs.writeFileSync(transcriptFilePath, transcription, 'utf8');

    // GCSにテキストアップロード
    const transcriptFileNameInGCS = fileNameInGCS.replace(/\.flac$/, '.txt');
    await storage.bucket(BUCKET_NAME).upload(transcriptFilePath, {
      destination: transcriptFileNameInGCS,
      metadata: { contentType: 'text/plain' }
    });
    console.log(`📝 テキストアップロード: gs://${BUCKET_NAME}/${transcriptFileNameInGCS}`);

    console.log("🧪 appendToSheet 呼び出し前:", { transcriptFileNameInGCS, transcription });

    // 認識に成功したときだけスプレッドシートに記録
    // ④ スプレッドシートに追記
    // ④ transcription が空でなければスプレッドシートに追記
    if (transcription !== '') {
    
     await appendToSheet({
       filename: fileNameInGCS,
       text: transcription
      });
    }else {
      console.log("🛑 スプレッドシートへの追記スキップ: 認識できず or 空文字");
    }

    


    // クリーンアップ
    fs.unlinkSync(inputPath);
    fs.unlinkSync(outputPath);

    // フロントに返却
    res.json({ text: transcription !== '' ? transcription : '音声が認識できませんでした。'
    });

  } catch (err) {
    console.error('❗処理失敗:', err);
    res.status(500).json({ error: 'STT processing error' });
  }
});

// 動作確認用
app.get('/', (req, res) => {
  //res.send('✅ Webhook STT サーバーが起動中');
  res.sendFile(path.join(__dirname, 'public', 'index-STT.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 サーバー起動: http://localhost:${PORT}`);
  console.log(`🚀 Listening on port ${PORT}`);
});

// CORS対応：全てのリクエストに CORS ヘッダーを追加
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*'); // 必要に応じて '*' を特定のドメインに
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  next();
});

