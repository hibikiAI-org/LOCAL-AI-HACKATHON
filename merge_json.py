import os
import json
import glob
import argparse

# コマンドライン引数のパーサーを作成
parser = argparse.ArgumentParser(description="Merge JSON files")
parser.add_argument("--input_dir", type=str, default=".", help="Directory containing input JSON files (default: current directory)")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the merged JSON file (default: current directory)")
args = parser.parse_args()

# 入力ディレクトリと出力ディレクトリを取得
input_dir = args.input_dir
output_dir = args.output_dir

# マージ対象のJSONファイルのパターンを指定
json_pattern = os.path.join(input_dir, "results_*.json")

# マージ後のJSONファイル名
merged_json_file = os.path.join(output_dir, "results.json")

# JSONファイルを検索
json_files = glob.glob(json_pattern)

# マージ後のデータを格納するリスト
merged_data = []

# 各JSONファイルを読み込んでマージ
for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
        merged_data.extend(data)

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# マージ後のデータをJSONファイルに書き込み
with open(merged_json_file, "w", encoding="utf-8") as file:
    json.dump(merged_data, file, ensure_ascii=False, indent=2)

print(f"JSONファイルのマージが完了しました。出力ファイル: {merged_json_file}")