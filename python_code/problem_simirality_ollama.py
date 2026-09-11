import pandas as pd
import torch
import re
import os
import random
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import requests # ★追加: API通信用

# --- 設定 ---
OLLAMA_API_URL = "http://127.0.0.1:11450/api/chat"
OLLAMA_MODEL = "deepseek-coder-v2" # 環境に合わせて変更

TARGET_FILES = ["/home/s22t324/project/result.csv"]
MAX_INPUT_CHARS = 8000 # ★トークン上限から文字数上限に変更
EVAL_SAMPLE_SIZE = 10
OUTPUT_FILE = "ollama_problem_result_K-means.csv"

# --- 1. データ準備 ---
print("=== データ読み込みと統合 ===")
df_list = []
for file_name in TARGET_FILES:
    if os.path.exists(file_name):
        try:
            # BOM付きUTF-8に対応
            _df = pd.read_csv(file_name, encoding='utf-8-sig')
            print(f"Loaded {file_name}: {_df.shape[0]} rows")
            df_list.append(_df)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    else:
        print(f"Warning: {file_name} not found.")

if not df_list:
    raise ValueError("読み込めるCSVファイルがありません。カレントディレクトリにファイルを配置してください。")

# 全データを統合
full_df = pd.concat(df_list, ignore_index=True)

# 必須カラムを original_code と generated_problem に指定
required_cols = ['original_code', 'generated_problem']
full_df = full_df.dropna(subset=required_cols)

# コードの特徴に基づく10件の抽出（K-Meansクラスタリング）
if len(full_df) > EVAL_SAMPLE_SIZE:
    print("\n=== コードの特徴に基づく10件の抽出（K-Meansクラスタリング） ===")
    # なでしこのソースコード（original_code）を特徴量抽出の対象とする
    texts = full_df['original_code'].fillna("").tolist()

    cluster_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    X = cluster_vectorizer.fit_transform(texts)

    num_clusters = EVAL_SAMPLE_SIZE
    kmeans = KMeans(n_clusters=num_clusters, random_state=123, n_init=10)
    full_df['cluster'] = kmeans.fit_predict(X)

    sampled_indices = []
    for i in range(num_clusters):
        cluster_data = full_df[full_df['cluster'] == i]
        if not cluster_data.empty:
            sampled_idx = cluster_data.sample(n=1, random_state=123).index[0]
            sampled_indices.append(sampled_idx)

    test_df = full_df.loc[sampled_indices].copy()
    
    print(f"Total Data: {len(full_df)} -> Validation Target (Clustered): {len(test_df)}")
    print("※各クラスタ（特徴グループ）から均等に1件ずつ抽出し、それ以外をFew-Shotのプールとして使用する。")
else:
    test_df = full_df
    print(f"Total Data: {len(full_df)} -> Validation Target: {len(test_df)} (All data used)")

# --- 2. ユーティリティ関数 ---
# ★変更: AutoModel などのロードを削除し、count_tokensをestimate_lengthに変更

def estimate_length(messages):
    """チャット形式のメッセージリストの総文字数を概算する"""
    return sum([len(m['content']) for m in messages])

# ★追加: 類似度計算用の中間Pythonコードを生成する関数
def generate_temp_python_code(problem_text):
    prompt = f"あなたはプログラマです。以下の問題文を解決するPython 3のコードのみを出力してください。説明やマークダウンは不要です。\n問題:\n{problem_text}"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    raw_text = response.json().get("message", {}).get("content", "")
    
    raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    clean_text = raw_text.replace("```python", "").replace("```", "").strip()
    return clean_text

# 修正後の関数定義
def create_text_to_code_messages(problem_text, train_pool, max_chars):
    """問題文からコードを生成する動的Few-Shotプロンプトを作成"""
    
    rule = (
        "以下の問題文の指示に従って、プログラミング言語「なでしこ」のコードを出力すること。\n"
        "【なでしこ記述時の絶対ルール】\n"
        "1. 助詞の重要性: 「て」「に」「を」「は」などの助詞が関数の引数を決定する極めて重要な役割を持つ。助詞は省略せず正確に記述すること。\n"
        "2. 語順: 日本語の自然な語順に従うこと（例: AにBを足す）。\n"
        "3. 制御構造: 「もし〜ならば」「〜の間」「〜回」などを正しく用いること。\n"
        "解説は一切不要である。コードのみを出力すること。"
    )
    
    # 対象の問題文
    base_messages = [{"role": "user", "content": f"{rule}\n問題:\n{problem_text}"}]
    
    # 例示候補をシャッフル（ここでは引数として受け取った train_pool を使用する）
    shots_messages = []
    shot_count = 0
    
    # ★修正箇所: shuffled_pool を train_pool に変更
    for _, row in train_pool.iterrows():
        new_shot = [
            {"role": "user", "content": f"{rule}\n問題:\n{row['generated_problem']}"},
            {"role": "assistant", "content": f"```nadesiko\n{row['original_code']}\n```"}
        ]
        
        potential_shots = shots_messages + new_shot
        full_messages_candidate = potential_shots + base_messages
        
        if estimate_length(full_messages_candidate) < max_chars:
            shots_messages = potential_shots
            shot_count += 1
        else:
            break
            
    return shots_messages + base_messages, shot_count

def extract_code_block(text):
    """生成テキストからコードブロックのみを抽出"""
    pattern = r"```(?:nadesiko)?\s*(.*)" 
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1)
        if "```" in content:
            content = content.split("```")[0]
        return content.strip()
    return re.sub(r'###.*', '', text).strip()

def normalize_text(text):
    """比較用に空白とハイフンを除去した文字列を返す"""
    text_no_hyphen = re.sub(r'-+', '', str(text))
    return "".join(text_no_hyphen.split())

def evaluate_non_blank_consistency(reference, candidate):
    """構造的一致の判定"""
    ref_clean = normalize_text(reference)
    cand_clean = normalize_text(candidate)

    is_perfect_match = 1.0 if ref_clean == cand_clean else 0.0
    similarity = SequenceMatcher(None, ref_clean, cand_clean).ratio()

    return is_perfect_match, similarity

# --- 4. 検証実行 ---
results = []
print(f"\n=== 検証開始 ({len(test_df)} items) ===")
total_shots_used = 0

# 類似度計算用のVectorizer（日本語の文字N-gramを使用。形態素解析不要で文脈の類似を捉えやすい）
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))

for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
    problem_text = row['generated_problem']
    correct_answer = row['original_code']
    
    # 自身をFew-Shotの候補（プール）から除外
    train_pool = full_df.drop(index).copy()
    
    # ターゲット問題文とプールの問題文を結合してTF-IDFを計算
    texts = [problem_text] + train_pool['generated_problem'].tolist()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # ターゲット問題（インデックス0）とそれ以外の問題のコサイン類似度を計算
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # 類似度スコアをプールに付与し、降順（似ている順）にソート
    train_pool['text_similarity'] = similarities
    sorted_pool = train_pool.sort_values(by='text_similarity', ascending=False)
    
    # なでしこ生成用のFew-Shotプロンプト作成
    messages, shot_num = create_text_to_code_messages(
        problem_text, sorted_pool, MAX_INPUT_CHARS
    )
    total_shots_used += shot_num
    
    try:
        # ★変更: ローカル推論からAPIリクエストに変更
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        raw_text = response.json().get("message", {}).get("content", "")
        
        # 最終出力からも思考プロセスを除去
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        cleaned_text = extract_code_block(raw_text)
        # ★ここまで
        
        corr_norm = normalize_text(correct_answer)
        gen_norm = normalize_text(cleaned_text)
        similarity_score = SequenceMatcher(None, corr_norm, gen_norm).ratio()

    except RuntimeError as e:
        if "out of memory" in str(e) or "CUDA" in str(e):
            print(f"\nError: GPU Memory overflow at index {index}. Skipping.")
        cleaned_text = "ERROR"
        similarity_score = 0.0
    finally:
        # GPUメモリの解放処理
        if 'inputs' in locals():
            del inputs
        if 'output_ids' in locals():
            del output_ids
        torch.cuda.empty_cache()

    results.append({
        "problem_text": problem_text,
        "correct_code": correct_answer,
        "generated_code": cleaned_text,
        "similarity": similarity_score,
        "shots_count": shot_num
    })

avg_shots = total_shots_used / len(test_df)
print(f"  -> Average Few-Shot count: {avg_shots:.1f}")

# =====================================================================

# --- 5. 結果集計と保存 ---
# （以降は元のコードのまま）
# --- 5. 結果集計と保存 ---
results_df = pd.DataFrame(results)

print("\n=== 検証結果サマリ ===")
print(f"Mean Similarity: {results_df['similarity'].mean():.4f}")

results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n詳細結果を {OUTPUT_FILE} に保存した。")