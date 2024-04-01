# local-ai-hackathon
[LOCAL AI HACKATHON #000](https://prtimes.jp/main/html/rd/p/000000007.000056944.html)で作成したスクリプトです。

# 環境構築
ryeを使ってPythonの環境構築を行います。
https://rye-up.com/guide/installation/

## src/local_ai_hackathon/yodas_cleansing.py
yodasから音声を取得、下記のフィルタリングをした後にrawフォルダとesd.listを作るスクリプト
- WADA SNR 100以上
- tarepan/SpeechMOS:v1.2.0のスコアが3以上

## run_parallel_jp.sh
Style-Bert-VITS2の並列学習を行うスクリプト
Style-Bert-VITS2のディレクトリ直下において　実行します
