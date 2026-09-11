import pandas as pd
import re
import os
import requests
from difflib import SequenceMatcher
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import AutoTokenizer

# --- 設定 ---
OLLAMA_API_URL = "http://127.0.0.1:11461/api/chat"
OLLAMA_MODEL = "deepseek-coder-v2" 
TARGET_FILES = ["structured_problems.csv"]

MAX_INPUT_TOKENS = 3500 
EVAL_SAMPLE_SIZE = 10

OUTPUT_FILE = "deepseek_structured_result_K-means.csv"

# --- 1. データ準備 ---
print("=== データ読み込みと統合 ===")
df_list = []
for file_name in TARGET_FILES:
    if os.path.exists(file_name):
        try:
            _df = pd.read_csv(file_name, encoding='utf-8-sig')
            print(f"Loaded {file_name}: {_df.shape[0]} rows")
            df_list.append(_df)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    else:
        print(f"Warning: {file_name} not found.")

if not df_list:
    raise ValueError("読み込めるCSVファイルがありません。カレントディレクトリにファイルを配置してください。")

full_df = pd.concat(df_list, ignore_index=True)

required_cols = ['original_code', 'generated_problem', 'structured_problem']
full_df = full_df.dropna(subset=required_cols)

if len(full_df) > EVAL_SAMPLE_SIZE:
    print("\n=== コードの特徴に基づく10件の抽出（K-Meansクラスタリング） ===")
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
    print("※各クラスタ（特徴グループ）から均等に1件ずつ抽出した。")
else:
    test_df = full_df
    print(f"Total Data: {len(full_df)} -> Validation Target: {len(test_df)} (All data used)")
    

# --- 2. モデル（Tokenizer）準備 ---
print("\n=== トークナイザーのロード中... ===")
# トークン数をElyzaと厳密に揃えるため、比較基準としてElyzaのTokenizerをカウント用に使用
tokenizer = AutoTokenizer.from_pretrained("elyza/Llama-3-ELYZA-JP-8B")


# --- 3. ユーティリティ関数 ---

def count_tokens(messages, tokenizer):
    """チャット形式のメッセージリストの総トークン数を概算する"""
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(text))
    except Exception:
        return sum([len(tokenizer.encode(m['content'])) for m in messages])

def create_text_to_code_messages(problem_text, train_pool, tokenizer, max_tokens, max_shots=10):
    """問題文からコードを生成するFew-Shotプロンプトを作成（JSONは探索にのみ使用するため含めない）"""
    
    rule = (
        "あなたは日本語プログラミング言語「なでしこ」の熟練エンジニアである。\n"
        "入力された問題文の要件を満たす、文法破綻のないなでしこコードを生成せよ。\n\n"
        "【厳守すべき重要ルール】\n"
        "1. 必ず[Few-shot Examples]として提示される例題の文法・書き方を厳密に真似て実装すること。\n"
        "2. なでしこ固有のブロック構造（「〜の間繰り返す」「●〜とは」「もし〜ならば〜違えば」「ここまで」等）を正しく用いること。\n"
        "3. 絶対に、Python等の他言語の構文（input(), rand(), ==, while, 括弧など）や、存在しない擬似コードを混入させないこと。\n"
        "4. 出力は純粋ななでしこコードのみとし、解説やMarkdown記法は一切不要である。"
    )
    
    sys_msg = [{"role": "system", "content": rule}]
    
    # 対象の問題（問題文のみ）
    target_messages = [{"role": "user", "content": f"問題文:\n{problem_text}"}]
    
    shots_messages = []
    shot_count = 0
    
    # 例示候補（探索済みのプール）から抽出
    for _, row in train_pool.iterrows():
        if shot_count >= max_shots:
            break
            
        # 例示も問題文と正解コードのみ
        new_shot = [
            {"role": "user", "content": f"問題文:\n{row['generated_problem']}"},
            {"role": "assistant", "content": f"```nadesiko\n{row['original_code']}\n```"}
        ]
        
        potential_shots = shots_messages + new_shot
        full_messages_candidate = sys_msg + potential_shots + target_messages
        
        if count_tokens(full_messages_candidate, tokenizer) < max_tokens:
            shots_messages = potential_shots
            shot_count += 1
        else:
            break
            
    return sys_msg + shots_messages + target_messages, shot_count


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

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))

for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
    problem_text = row['generated_problem']
    correct_answer = row['original_code']
    structured_target = row['structured_problem']
    
    # 自身をFew-Shotの候補（プール）から除外
    train_pool = full_df.drop(index).copy()
    
    # ★探索：生成した構造化テキスト(JSON)間で類似度計算を行う
    texts = [structured_target] + train_pool['structured_problem'].tolist()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    train_pool['structure_similarity'] = similarities
    sorted_pool = train_pool.sort_values(by='structure_similarity', ascending=False)
    
    # プロンプトには「問題文」のみを渡し、例示には探索済みのsorted_poolを使用する
    messages, shot_num = create_text_to_code_messages(
        problem_text, sorted_pool, tokenizer, MAX_INPUT_TOKENS, max_shots=10
    )
    total_shots_used += shot_num
    
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        raw_text = response.json().get("message", {}).get("content", "")
        
        # DeepSeek特有の思考プロセスを除去
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        cleaned_text = extract_code_block(raw_text)
        
        corr_norm = normalize_text(correct_answer)
        gen_norm = normalize_text(cleaned_text)
        similarity_score = SequenceMatcher(None, corr_norm, gen_norm).ratio()
        
    except Exception as e:
        print(f"\nError at index {index}: {e}")
        cleaned_text = "ERROR"
        similarity_score = 0.0

    results.append({
        "problem_text": problem_text,
        "correct_code": correct_answer,
        "generated_code": cleaned_text,
        "structured_problem": structured_target, # 記録用としてCSVにはJSONを残す
        "similarity": similarity_score,
        "shots_count": shot_num
    })

avg_shots = total_shots_used / len(test_df)
print(f"  -> Average Few-Shot count: {avg_shots:.1f}")

# --- 5. 結果集計と保存 ---
results_df = pd.DataFrame(results)

print("\n=== 検証結果サマリ ===")
print(f"Mean Similarity: {results_df['similarity'].mean():.4f}")

results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n詳細結果を {OUTPUT_FILE} に保存した。")