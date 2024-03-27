import os
import torch
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
import numpy as np
import whisper
import argparse
import multiprocessing as mp
import logging
import json
import shutil
import glob

# ロガーの作成
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ログのフォーマットを設定
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# コンソールハンドラを作成し、ロガーに追加
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# データを前処理するための関数
def preprocess_audio(data):
    # データが整数型の場合、浮動小数点型に変換
    if data.dtype == np.int16:
        data = data.astype(np.float32) / np.iinfo(np.int16).max
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / np.iinfo(np.int32).max

    # ステレオをモノラルに変換（必要があれば）
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    return data

def wada_snr(wav):
    # Direct blind estimation of the SNR of a speech signal.
    #
    # Paper on WADA SNR:
    #   http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    #
    # This function was adapted from this matlab code:
    #   https://labrosa.ee.columbia.edu/projects/snreval/#9

    # init
    eps = 1e-10
    # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
    db_vals = np.arange(-20, 101)
    g_vals = np.array([0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192 , 0.42077252, 0.42452799, 0.42918886, 0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236, 0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349 , 1.01047155, 1.0362095 , 1.06136425, 1.08579312, 1.1094819 , 1.13277995, 1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935, 1.3605727 , 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938, 1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 1.5229097 , 1.528578  , 1.53389835, 1.5391211 , 1.5439065 , 1.54858517, 1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767, 1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477, 1.5941969 , 1.59693155, 1.599446  , 1.60185011, 1.60408668, 1.60627134, 1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143, 1.62632463, 1.6274027 , 1.62842767, 1.62945532, 1.6303307 , 1.63128026, 1.63204102])

    # peak normalize, get magnitude, clip lower bound
    wav = np.array(wav)
    wav = wav / abs(wav).max()
    abs_wav = abs(wav)
    abs_wav[abs_wav < eps] = eps

    # calcuate statistics
    # E[|z|]
    v1 = max(eps, abs_wav.mean())
    # E[log|z|]
    v2 = np.log(abs_wav).mean()
    # log(E[|z|]) - E[log(|z|)]
    v3 = np.log(v1) - v2

    # table interpolation
    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    # handle edge cases or interpolate
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
    else:
        wav_snr = db_vals[wav_snr_idx] + \
            (v3-g_vals[wav_snr_idx]) / (g_vals[wav_snr_idx+1] - \
            g_vals[wav_snr_idx]) * (db_vals[wav_snr_idx+1] - db_vals[wav_snr_idx])

    # Calculate SNR
    dEng = sum(wav**2)
    dFactor = 10**(wav_snr / 10)
    dNoiseEng = dEng / (1 + dFactor) # Noise energy
    dSigEng = dEng * dFactor / (1 + dFactor) # Signal energy
    snr = 10 * np.log10(dSigEng / dNoiseEng)

    return snr

def analyze_audio_data(gpu_id, data_partition, whisper_model, snr_threshold, score_threshold, result_queue):
    results = []
    for item in data_partition:
        result = process_audio_data(item, gpu_id=gpu_id, whisper_model=whisper_model, snr_threshold=snr_threshold, score_threshold=score_threshold)
        if result:
            results.append(result)
    result_queue.put(results)

def process_audio_data(item, gpu_id, whisper_model, snr_threshold, score_threshold):
    # logger.info(f"{item.get(id)}", end=",")
    logger.debug(item)
    if not isinstance(item, dict):
        logger.info(f"無効なデータ形式: {item}")
        return None

    # 音声データを読み込む
    audio_data = item['audio']['array']

    # speech-mosを使用して数値を取得
    torch.cuda.set_device(gpu_id)
    with torch.no_grad():
        device = torch.device(f"cuda:{gpu_id}")
        # audio_data = torch.from_numpy(audio_data).unsqueeze(0).to(device)
        # speech-mosの予測器を初期化
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        predictor = predictor.to(device)
        # サンプリングレートを取得または設定
        sr = item['audio'].get('sampling_rate', 16000)
        # データを前処理
        preprocess_audio_data = preprocess_audio(audio_data)

        snr = wada_snr(preprocess_audio_data)
        
        if snr_threshold > snr or np.isnan(snr):
            return None
        
        # min_length = 10  # モデルの要件に基づいて、この値を調整してください
        # if len(audio_data) < min_length:
        #     logger.info(f"長さ {len(audio_data)} のオーディオデータをスキップします（最小必要長: {min_length}）")
        #     return None
        # audio_data_tensor = audio_data.unsqueeze(0).to(torch.float32)
        audio_data_tensor = torch.from_numpy(preprocess_audio_data).unsqueeze(0).to(torch.float32).to(device)
        score = predictor(audio_data_tensor.to(torch.float32), sr) 

        del predictor
        del audio_data_tensor
        if score_threshold > score:
            logger.debug(f"skip {score_threshold} > {score}")
            return


    # データからidを取得
    uuid = item['utt_id']

    # whisperを使用して文字起こしを行う
    whisper_model = whisper_model.to(device)
    audio_data = audio_data.astype(np.float32)
    transcription = whisper_model.transcribe(audio_data, language='ja')['text']

    # 結果を指定された形式の文字列に整形
    # result = f"{uuid}|pretraing1|JP|{transcription}"

     # 結果を辞書形式で整形
    
    result = {
        'id': item['id'],
        'uuid': uuid,
        'snr': float(snr),
        'score': float(score),
        'transcription': transcription,
        'source_transcription': item['text'],
        'path': item['audio']['path']
    }

    logger.info(f"result {result}")


    return result

def process_results(ds, whisper_model, snr_threshold, score_threshold, start, end, data_dir):
    num_gpus = torch.cuda.device_count()
    data_partitions = np.array_split(ds, num_gpus)

    processes = []
    result_queue = mp.Queue()
    for gpu_id in range(num_gpus):
        p = mp.Process(target=analyze_audio_data, args=(gpu_id, data_partitions[gpu_id], whisper_model, snr_threshold, score_threshold, result_queue))
        p.start()
        processes.append(p)

    all_results = []
    for p in processes:
        p.join()

    while not result_queue.empty():
        results = result_queue.get()
        if results:
            all_results.extend(results)

    # wavディレクトリを作成
    wav_dir = os.path.join(data_dir, 'raw')
    os.makedirs(wav_dir, exist_ok=True)

    with open(os.path.join(data_dir, f'esd_{start}-{end}.list'), 'w', encoding='utf-8') as f:
        for result in all_results:
            uuid = result["uuid"]
            f.write(f"{os.path.join(wav_dir, f'{uuid}.wav')}|pretraing1|JP|{result['transcription']}\n")
            src_path = result['path']
            dst_path = os.path.join(wav_dir, f"{uuid}.wav")
            shutil.copy(src_path, dst_path)

    with open(os.path.join(data_dir, f'results_{start}-{end}.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description='Audio analysis and transcription')
    parser.add_argument('--start', type=int, default=0, help='Starting index of the dataset (default: 0)')
    parser.add_argument('--end', type=int, default=None, help='Ending index of the dataset (default: None, process all items)')
    parser.add_argument('--snr_threshold', type=float, default=100.0, help='SNR threshold for filtering (default: 100.0)')
    parser.add_argument('--score_threshold', type=float, default=3.0, help='Score threshold for filtering (default: 3.0)')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for processing (default: 1000)')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to store output data (default: data)')

    args = parser.parse_args()

    # GPUが利用可能な場合は、GPUを使用するように設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # デバイス情報をログ出力
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        logger.info(f"使用可能なGPU数: {num_gpus}")
        logger.info(f"メインGPU: {torch.cuda.get_device_name(device)}")
    else:
        logger.info("CPUを使用します")

    # whisperモデルを初期化
    whisper_model = whisper.load_model("large")
    mp.set_start_method('spawn')

    logger.info(f"data_dir {args.data_dir}")
    # データセットの読み込み
    if args.end is None:
        ds0 = load_dataset('espnet/yodas', 'ja000', split=f'train[{args.start}:]')
    else:
        ds0 = load_dataset('espnet/yodas', 'ja000', split=f'train[{args.start}:{args.end}]')
    logger.info(f"データ数: {len(ds0)}")

    # バッチ処理
    batch_size = args.batch_size
    for batch_start in range(args.start, len(ds0), batch_size):
        batch_end = min(batch_start + batch_size - 1, len(ds0))
        batch_ds = load_dataset('espnet/yodas', 'ja000', split=f'train[{batch_start}:{batch_end}]')

        logger.info(f"Processing batch: {batch_start} - {batch_end}")

        # 分析を実行
        process_results(batch_ds, whisper_model=whisper_model, snr_threshold=args.snr_threshold, score_threshold=args.score_threshold, start=batch_start, end=batch_end, data_dir=args.data_dir)


     # 全てのesd.listをマージ
    esd_files = glob.glob(os.path.join(args.data_dir, 'esd_*.list'))
    with open(os.path.join(args.data_dir, 'esd.list'), 'w', encoding='utf-8') as outfile:
        for esd_file in esd_files:
            with open(esd_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

    logger.info(f"All esd.list files merged into {os.path.join(args.data_dir, 'esd.list')}")