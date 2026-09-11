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

# ★検証するShot数のリスト（0はZero-Shot、それ以外はFew-Shot）
SHOT_COUNTS_TO_TEST = [0, 1, 2, 4, 8]

# 検証データの件数（各Shot数ごとにこの件数を回します）
EVAL_SAMPLE_SIZE = 10

# 出力ファイル名
OUTPUT_FILE = "few_shot_scaling_result.csv"

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

if not df_list:
    raise ValueError("CSVファイルが見つかりません。")

full_df = pd.concat(df_list, ignore_index=True)
required_cols = ['original_code', 'branks_level1', 'branks_level2', 'branks_level3', 'branks_level4', 'branks_level5']
full_df = full_df.dropna(subset=required_cols)

# 検証用データの抽出
if len(full_df) > EVAL_SAMPLE_SIZE:
    test_df = full_df.sample(n=EVAL_SAMPLE_SIZE, random_state=42)
    print(f"Validation Target: {len(test_df)} items (from {len(full_df)} total)")
else:
    test_df = full_df
    print(f"Validation Target: {len(test_df)} items (All data)")

# --- 2. モデル準備 ---
print("\n=== モデルロード中... ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

# --- 3. ユーティリティ関数 ---

def create_chat_messages(target_code, level, train_pool, n_shots):
    """指定されたショット数(n_shots)でプロンプトを作成"""
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
    
    messages = []
    
    # Few-Shot (n_shots > 0 の場合のみ)
    if n_shots > 0 and len(train_pool) > 0:
        # 指定数だけランダムに抽出
        # ※ データ数が足りない場合はあるだけ使う
        sample_count = min(len(train_pool), n_shots)
        shots = train_pool.sample(n=sample_count) # ランダム抽出
        
        for _, row in shots.iterrows():
            messages.append({"role": "user", "content": f"{rule}\nコード:\n{row['original_code']}"})
            messages.append({"role": "assistant", "content": f"```nadesiko\n{row[target_col]}\n```"})
    
    # Target
    messages.append({"role": "user", "content": f"{rule}\nコード:\n{target_code}"})
    return messages

def extract_code_block(text):
    pattern = r"```(?:nadesiko)?\s*(.*)" 
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1)
        if "```" in content:
            content = content.split("```")[0]
        return content.strip()
    return re.sub(r'###.*', '', text).strip()

def normalize_text(text):
    text_no_hyphen = re.sub(r'-+', '', str(text))
    return "".join(text_no_hyphen.split())

def evaluate_non_blank_consistency(reference, candidate):
    ref_clean = normalize_text(reference)
    cand_clean = normalize_text(candidate)
    is_perfect_match = 1.0 if ref_clean == cand_clean else 0.0
    similarity = SequenceMatcher(None, ref_clean, cand_clean).ratio()
    return is_perfect_match, similarity

# --- 4. 検証実行 ---
results = []
print(f"\n=== 検証開始 (Shots: {SHOT_COUNTS_TO_TEST}) ===")

# Shot数のループ
for n_shots in SHOT_COUNTS_TO_TEST:
    print(f"\n>>> Testing with n_shots = {n_shots} <<<")
    
    # レベルのループ
    for level in [1, 2, 3, 4, 5]:
        category_char = 'ABCDE'[level-1]
        target_col = f'branks_level{level}'
        
        # データのループ
        # tqdmで進捗表示
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"L{level}-Shot{n_shots}"):
            original_code = row['original_code']
            correct_answer = row[target_col]
            
            # 自分自身を除くプールを作成
            train_pool = full_df.drop(index)
            
            # メッセージ作成
            messages = create_chat_messages(original_code, level, train_pool, n_shots)
            
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
                
                # 評価
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

            except Exception as e:
                print(f"Error at Shot{n_shots}-L{level}-{index}: {e}")
                cleaned_text = "ERROR"
                match_score = 0.0
                similarity_score = 0.0

            results.append({
                "n_shots": n_shots, # ショット数を記録
                "level": level,
                "category": category_char,
                "original_code": original_code,
                "correct_answer": correct_answer,
                "generated_answer": cleaned_text,
                "structure_match": match_score, 
                "similarity": similarity_score
            })

# --- 5. 結果集計と保存 ---
results_df = pd.DataFrame(results)

print("\n=== ショット数別 結果サマリ ===")
# Shot数ごとの平均スコアを表示
summary = results_df.groupby(['n_shots'])[['structure_match', 'similarity']].mean()
print(summary)

results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n詳細結果を {OUTPUT_FILE} に保存しました。")