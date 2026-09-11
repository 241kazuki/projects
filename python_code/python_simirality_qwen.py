import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-8B"
PEFT_MODEL_ID = "lorenzocazzador/coder-grpo-qwen3-8b"
TARGET_FILES = [
    "add_python_result_qwen3.csv",
    "add_python_result_deepseek.csv",
    "add_python_result.csv"
]

MAX_INPUT_TOKENS = 3500
EVAL_SAMPLE_SIZE = 10
OUTPUT_FILE = "qwen_grpo_python_result_K-means.csv"

print("=== データ読み込みと統合 ===")
df_list = []
for file_name in TARGET_FILES:
    if os.path.exists(file_name):
        try:
            _df = pd.read_csv(file_name, encoding='utf-8-sig')
            print(f"Loaded {file_name}: {_df.shape[0]} rows")
            df_list.append(_df)
            break  # 最初に存在するファイルを優先読み込み
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    else:
        print(f"Warning: {file_name} not found.")

if not df_list:
    raise ValueError("読み込めるCSVファイルがありません。パスを確認してください。")

full_df = pd.concat(df_list, ignore_index=True)
required_cols = ['original_code', 'generated_problem', 'python_code']
full_df = full_df.dropna(subset=required_cols).reset_index(drop=True)

# K-Meansクラスタリングによる10件の検証用サンプル選出
if len(full_df) > EVAL_SAMPLE_SIZE:
    print("\n=== Pythonコードの特徴に基づく10件の抽出（K-Meansクラスタリング） ===")
    texts = full_df['python_code'].fillna("").tolist()

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
else:
    test_df = full_df
    print(f"Total Data: {len(full_df)} -> Validation Target: {len(test_df)} (All data used)")

print("\n=== モデルおよびPEFTアダプターのロード中... ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL_ID, trust_remote_code=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

print(f"=== PEFTアダプター {PEFT_MODEL_ID} を統合中 ===")
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)

# Qwen3のチャットテンプレートが enable_thinking をサポートしているか確認するための
# 安全なラッパー（未対応テンプレートでもエラーにならないようにする）
def apply_chat_template_safe(messages, tokenizer, add_generation_prompt=True):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,  # ← ここが今回の主な修正点：思考モードを明示的にOFF
        )
    except TypeError:
        # enable_thinking引数に対応していない場合のフォールバック
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

def count_tokens(messages, tokenizer):
    """メッセージリストの総トークン数を計算する"""
    try:
        text = apply_chat_template_safe(messages, tokenizer)
        return len(tokenizer.encode(text))
    except Exception:
        return sum([len(tokenizer.encode(m['content'])) for m in messages])

def generate_temp_python_code(problem_text, tokenizer, model):
    """問題文から一時的なPythonコードを生成する（類似度計算用）"""
    sys_msg = "あなたはプログラマである。入力された問題文を解決するPython 3のコードのみを出力せよ。説明やコメント、マークダウンは一切不要である。"
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"問題:\n{problem_text}"}
    ]

    formatted_prompt = apply_chat_template_safe(messages, tokenizer)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        raw_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        clean_text = strip_thinking(raw_text)
        clean_text = clean_text.replace("```python", "").replace("```", "").strip()
        return clean_text
    finally:
        del inputs
        if 'output_ids' in locals():
            del output_ids
        torch.cuda.empty_cache()

def create_text_to_code_messages(problem_text, train_pool, tokenizer, max_tokens, max_shots=10):
    """問題文から「なでしこ」コードを生成する動的Few-Shotプロンプトを作成"""
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
    target_messages = [{"role": "user", "content": f"問題文:\n{problem_text}"}]

    shots_messages = []
    shot_count = 0

    for _, row in train_pool.iterrows():
        if shot_count >= max_shots:
            break

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

def strip_thinking(text):
    """<think>ブロックを除去する。閉じタグがない（トークン数不足で途中で
    切れた）場合は、思考の残骸が本文に混入しないよう全体を空文字にする。"""
    if "<think>" in text and "</think>" not in text:
        # 生成が思考の途中で打ち切られた = 有効なコードは得られていない
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def extract_code_block(text):
    """生成テキストからコードブロックのみを抽出"""
    text = strip_thinking(text)
    if not text:
        return ""
    pattern = r"```(?:nadesiko)?\s*(.*?)(?:```|$)"
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

results = []
print(f"\n=== 検証開始 ({len(test_df)} items) ===")
total_shots_used = 0

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))

for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Qwen-GRPO (Pythonベース) 評価中"):
    problem_text = row['generated_problem']
    correct_answer = row['original_code']

    # 1. 対象問題に対するPythonコードを一時生成
    temp_python_code = generate_temp_python_code(problem_text, tokenizer, model)

    train_pool = full_df.drop(index).copy()

    # 2. 一時生成したPythonコードとプール内のPythonコード間で類似度計算
    texts = [temp_python_code] + train_pool['python_code'].tolist()
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    train_pool['python_similarity'] = similarities
    sorted_pool = train_pool.sort_values(by='python_similarity', ascending=False)

    messages, shot_num = create_text_to_code_messages(
        problem_text, sorted_pool, tokenizer, MAX_INPUT_TOKENS, max_shots=10
    )
    total_shots_used += shot_num

    try:
        formatted_prompt = apply_chat_template_safe(messages, tokenizer)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        raw_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        cleaned_text = extract_code_block(raw_text)

        if not cleaned_text:
            # 思考が閉じられず有効なコードが得られなかったケース
            cleaned_text = "GENERATION_INCOMPLETE"
            similarity_score = 0.0
        else:
            corr_norm = normalize_text(correct_answer)
            gen_norm = normalize_text(cleaned_text)
            similarity_score = SequenceMatcher(None, corr_norm, gen_norm).ratio()

    except Exception as e:
        print(f"\nError at index {index}: {e}")
        cleaned_text = "ERROR"
        similarity_score = 0.0
    finally:
        if 'inputs' in locals():
            del inputs
        if 'output_ids' in locals():
            del output_ids
        torch.cuda.empty_cache()

    results.append({
        "problem_text": problem_text,
        "correct_code": correct_answer,
        "generated_code": cleaned_text,
        "temp_python_code": temp_python_code,
        "similarity": similarity_score,
        "shots_count": shot_num
    })

avg_shots = total_shots_used / len(test_df)
print(f"  -> Average Few-Shot count: {avg_shots:.1f}")

results_df = pd.DataFrame(results)

print("\n=== 検証結果サマリ ===")
print(f"Mean Similarity: {results_df['similarity'].mean():.4f}")
incomplete_count = (results_df['generated_code'] == 'GENERATION_INCOMPLETE').sum()
if incomplete_count:
    print(f"注意: {incomplete_count} 件で思考が完了せずコードが得られませんでした（max_new_tokens不足の可能性）。")

results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n詳細結果を {OUTPUT_FILE} に保存しました。")