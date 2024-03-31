import os
import json
import wave
import sys

def get_total_duration(wav_folder):
    total_duration = 0
    for file in os.listdir(wav_folder):
        if file.endswith(".wav"):
            wav_path = os.path.join(wav_folder, file)
            with wave.open(wav_path, 'r') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                total_duration += duration
    return total_duration

def format_duration(duration):
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60
    return f"{hours}時間{minutes}分{seconds:.2f}秒"

def main(wav_folder):
    total_duration = get_total_duration(wav_folder)
    formatted_duration = format_duration(total_duration)
    output = {
        "total_seconds": total_duration,
        "formatted_duration": formatted_duration
    }
    with open("duration.json", "w") as json_file:
        json.dump(output, json_file, indent=2)
    print(f"合計時間: {formatted_duration}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python script.py <wavフォルダのパス>")
        sys.exit(1)
    wav_folder = sys.argv[1]
    main(wav_folder)