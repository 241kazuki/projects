import pandas as pd
import torch
import re
import os
import random
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 設定 ---
MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"
TARGET_FILES = ['output_ch1.csv', 'output_ch2.csv', 'output_ch3.csv', 'output_ch4.csv']

# ★プロンプトに入力する最大トークン数の目安
# エラーが出る場合はこの値を小さくしてください（例: 2048, 3072...）
MAX_INPUT_TOKENS = 3500 

# ★変更: 検証データの件数をZero-Shot版に合わせて「10件」に設定
EVAL_SAMPLE_SIZE = 10

# 出力ファイル名
OUTPUT_FILE = "category_validation_result.csv"

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

# 必要なカラムの確認と欠損値処理
required_cols = ['original_code', 'branks_level1', 'branks_level2', 'branks_level3', 'branks_level4', 'branks_level5']
full_df = full_df.dropna(subset=required_cols)

# 検証用データの抽出（全データから10件）
if len(full_df) > EVAL_SAMPLE_SIZE:
    test_df = full_df.sample(n=EVAL_SAMPLE_SIZE, random_state=42)
    print(f"Total Data: {len(full_df)} -> Validation Target: {len(test_df)}")
    print("※検証対象以外のデータを、トークン上限まで可能な限り例示(Few-Shot)として使用します。")
else:
    test_df = full_df
    print(f"Total Data: {len(full_df)} -> Validation Target: {len(test_df)} (All data used)")

# --- 2. モデル準備 ---
print("\n=== モデルロード中... ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

# --- 3. ユーティリティ関数 ---

def count_tokens(messages, tokenizer):
    """チャット形式のメッセージリストの総トークン数を概算する"""
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(text))
    except Exception:
        return sum([len(tokenizer.encode(m['content'])) for m in messages])

def create_chat_messages_dynamic(target_code, level, train_pool, tokenizer, max_tokens):
    """
    トークン数がmax_tokensに収まる範囲で、train_poolから例示を詰め込む
    """
    rules_text = {
        1: """【タスク: カテゴリA（データ・演算）に基づく空欄化】
以下の要素を「---」に置換してください。それ以外は変更しないでください。
1. 値: 数値、文字列の中身
2. 演算子: ＝ ＋ － × ÷ など""",
        2: """【タスク: カテゴリB（基本命令）に基づく空欄化】
以下の要素を「---」に置換してください。それ以外は変更しないでください。
1. 入出力命令: 表示、言う、尋ねる など
2. 変数操作: 代入""",
        3: """【タスク: カテゴリC（構造・ロジック）に基づく空欄化】
以下の制御構文に関わる語句を「---」に置換してください。それ以外は変更しないでください。
1. 繰り返し制御: 繰り返す、間、反復、抜ける など
2. 条件分岐: もし、ならば、違えば、ここまで など""",
        4: """【タスク: カテゴリD（応用・データ操作）に基づく空欄化】
以下の高度なデータ操作に関わる語句を「---」に置換してください。それ以外は変更しないでください。
1. 配列・辞書操作
2. 数値・データ保存（乱数、保存など）
3. その他: 関数、それ、対象、JSON、CSV""",
        5: """【タスク: カテゴリE（ドメイン固有）に基づく空欄化】
以下の特定用途向け命令を「---」に置換してください。それ以外は変更しないでください。
1. 描画・GUI: カメ〜、描画、ボタン
2. インタラクション: クリック
3. 通信・その他: 通信、グラフ、Ajax"""
    }
    
    rule = rules_text.get(level, "コードの一部をルールに基づいて隠蔽してください。")
    target_col = f'branks_level{level}'
    
    # ベースメッセージ（ルール + 対象コード）
    base_messages = [{"role": "user", "content": f"{rule}\nコード:\n{target_code}"}]
    
    # 例示候補をシャッフル
    shuffled_pool = train_pool.sample(frac=1, random_state=random.randint(0, 10000))
    
    shots_messages = []
    
    # 制限に達するまで例示を追加
    shot_count = 0
    for _, row in shuffled_pool.iterrows():
        new_shot = [
            {"role": "user", "content": f"{rule}\nコード:\n{row['original_code']}"},
            {"role": "assistant", "content": f"```nadesiko\n{row[target_col]}\n```"}
        ]
        
        potential_shots = shots_messages + new_shot
        full_messages_candidate = potential_shots + base_messages
        
        token_count = count_tokens(full_messages_candidate, tokenizer)
        
        if token_count < max_tokens:
            shots_messages = potential_shots
            shot_count += 1
        else:
            break
            
    final_messages = shots_messages + base_messages
    return final_messages, shot_count

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
print(f"\n=== 検証開始 (Level 1~5 / Category A~E / {len(test_df)} items) ===")
print(f"Max Input Tokens: {MAX_INPUT_TOKENS}")

for level in [1, 2, 3, 4, 5]:
    category_char = 'ABCDE'[level-1]
    print(f"\n--- Level {level} (Category {category_char}) Evaluating... ---")
    
    total_shots_used = 0
    
    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        original_code = row['original_code']
        correct_answer = row[f'branks_level{level}']
        
        # 自分自身(index)を除く全データを例示候補とする
        train_pool = full_df.drop(index)
        
        # 動的Few-Shot作成
        messages, shot_num = create_chat_messages_dynamic(
            original_code, level, train_pool, tokenizer, MAX_INPUT_TOKENS
        )
        total_shots_used += shot_num
        
        try:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                )
            
            raw_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            cleaned_text = extract_code_block(raw_text)
            
            orig_norm = "".join(str(original_code).split())
            corr_hyphen_removed = normalize_text(correct_answer)
            gen_hyphen_removed = normalize_text(cleaned_text)
            
            is_correct_no_change = (orig_norm == corr_hyphen_removed)
            is_gen_no_change = (orig_norm == gen_hyphen_removed)
            
            if is_correct_no_change and is_gen_no_change:
                match_score = None
                similarity_score = None
            else:
                match_score, similarity_score = evaluate_non_blank_consistency(correct_answer, cleaned_text)
                
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                print(f"\nError: GPU Memory overflow at index {index}. Skipping.")
                torch.cuda.empty_cache()
            cleaned_text = "ERROR"
            match_score = 0.0
            similarity_score = 0.0

        results.append({
            "level": level,
            "category": category_char,
            "original_code": original_code,
            "correct_answer": correct_answer,
            "generated_answer": cleaned_text,
            "structure_match": match_score, 
            "similarity": similarity_score,
            "shots_count": shot_num
        })
    
    avg_shots = total_shots_used / len(test_df)
    print(f"  -> Average Few-Shot count: {avg_shots:.1f}")

# --- 5. 結果集計と保存 ---
results_df = pd.DataFrame(results)

print("\n=== カテゴリ別 検証結果サマリ ===")
summary = results_df.groupby(['level', 'category'])[['structure_match', 'similarity', 'shots_count']].mean()
print(summary)

results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n詳細結果を {OUTPUT_FILE} に保存しました。")