import pandas as pd
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 設定 ---
MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"

# 学習用データ（なでしこ）
TRAIN_FILES = ['output_ch1.csv', 'output_ch2.csv', 'output_ch3.csv', 'output_ch4.csv']

# 検証用データ（前回の生成結果ファイルを使用）
# ここに含まれる original_code を対象に検証を行います
TEST_FILE = "produire_generation_sample.csv" 

# Few-Shot設定（なでしこの例を何個見せるか）
FEW_SHOT_COUNT = 3

# 出力ファイル名
OUTPUT_FILE = "produire_verification_with_nadesiko_learning.csv"

# --- 1. データ準備 ---
print("=== データ読み込み ===")

# 学習データ（なでしこ）の読み込み
train_df_list = []
for f in TRAIN_FILES:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            train_df_list.append(df)
        except Exception as e:
            print(f"Warning: {f} could not be loaded: {e}")

if not train_df_list:
    raise ValueError("学習用データ（なでしこCSV）が見つかりません。")
train_df = pd.concat(train_df_list, ignore_index=True)
print(f"学習データ(なでしこ): {len(train_df)} 件")

# 必要なカラム（branks_levelX）の確認
required_cols = ['original_code', 'branks_level1', 'branks_level2', 'branks_level3', 'branks_level4', 'branks_level5']
train_df = train_df.dropna(subset=required_cols)

# 検証データ（プロデル・固定対象）の読み込み
if os.path.exists(TEST_FILE):
    try:
        # 前回の結果CSVを読み込む
        sample_df = pd.read_csv(TEST_FILE, encoding='utf-8-sig')
        
        # 'file_name' と 'original_code' の組み合わせで重複を除去し、検証対象リストを作成
        # これにより、前回使用されたコード一式を特定します
        if 'original_code' in sample_df.columns:
            test_df = sample_df[['file_name', 'original_code']].drop_duplicates()
            print(f"検証データ(プロデル): {len(test_df)} 件 (ファイル {TEST_FILE} から抽出)")
        else:
            raise ValueError(f"{TEST_FILE} に 'original_code' カラムがありません。")
            
    except Exception as e:
        print(f"Error loading {TEST_FILE}: {e}")
        exit()
else:
    print(f"Error: {TEST_FILE} が見つかりません。先にサンプル生成を行ってください。")
    exit()

# --- 2. モデル準備 ---
print("\n=== モデルロード中... ===")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
except Exception as e:
    print(f"モデルロードエラー: {e}")
    exit()

# --- 3. プロンプト作成関数（クロス言語Few-Shot） ---

def create_chat_messages(target_code, level, train_data, shot_count=3):
    """
    なでしこのデータを例示（Few-Shot）として使用し、
    プロデルのコードを変換させるプロンプトを作成する。
    """
    
    # プロデル用のタスク定義
    rules_text = {
        1: """【タスク: カテゴリA（データ・演算）の隠蔽】
コード内の「数値」「文字列」「演算子」を「---」に置換してください。""",
        
        2: """【タスク: カテゴリB（基本命令）の隠蔽】
コード内の「入出力命令（表示など）」「変数操作（代入など）」を「---」に置換してください。""",
        
        3: """【タスク: カテゴリC（構造・ロジック）の隠蔽】
コード内の「繰り返し処理」「条件分岐（もし〜なら）」を「---」に置換してください。""",
        
        4: """【タスク: カテゴリD（応用・データ操作）の隠蔽】
コード内の「配列・辞書操作」「関数/手順定義」「ファイル操作」を「---」に置換してください。""",
        
        5: """【タスク: カテゴリE（ドメイン固有）の隠蔽】
コード内の「GUI部品」「イベント処理」「専用コマンド」を「---」に置換してください。"""
    }
    
    rule = rules_text.get(level, "コードの一部をルールに基づいて隠蔽してください。")
    target_col = f'branks_level{level}'
    
    messages = []
    
    # Few-Shot: なでしこのデータを例示として追加
    if shot_count > 0 and len(train_data) > 0:
        shots = train_data.sample(n=min(len(train_data), shot_count))
        for _, row in shots.iterrows():
            messages.append({
                "role": "user", 
                "content": f"{rule}\nコード:\n{row['original_code']}"
            })
            messages.append({
                "role": "assistant", 
                "content": f"```nadesiko\n{row[target_col]}\n```"
            })
    
    # Target: プロデルのコード
    messages.append({
        "role": "user", 
        "content": f"{rule}\nコード:\n{target_code}"
    })
    
    return messages

def extract_code_block(text):
    """生成テキストからコードブロックのみを抽出"""
    pattern = r"```(?:produire|nadesiko|plain)?\s*(.*)" 
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1)
        if "```" in content:
            content = content.split("```")[0]
        return content.strip()
    return re.sub(r'###.*', '', text).strip()

# --- 4. 生成実行 ---
results = []
print(f"\n=== 検証開始 (なでしこ学習 -> プロデル検証) ===")

for level in [1, 2, 3, 4, 5]:
    category_char = 'ABCDE'[level-1]
    print(f"\n--- Level {level} (Category {category_char}) Evaluating... ---")
    
    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        original_code = row['original_code']
        # 前回のファイル名を引き継ぐ（カラムがあれば）
        file_name = row.get('file_name', f'code_{index}')
        
        # プロンプト作成（なでしこ学習データを使用）
        messages = create_chat_messages(original_code, level, train_df, shot_count=FEW_SHOT_COUNT)
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
        
        results.append({
            "file_name": file_name,
            "level": level,
            "category": category_char,
            "learning_source": "Nadesiko(Few-Shot)",
            "original_code": original_code,
            "generated_answer": cleaned_text
        })

# --- 5. 結果保存 ---
results_df = pd.DataFrame(results)

# 見やすさのためにソート
results_df = results_df.sort_values(by=['file_name', 'level'])

results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n検証完了。結果を {OUTPUT_FILE} に保存しました。")