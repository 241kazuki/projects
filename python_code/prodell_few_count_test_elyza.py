import pandas as pd
import torch
import re
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 設定 ---
MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"

# 例示（Few-Shot）の参照元ファイル
TRAIN_FILES = ['output_ch1.csv', 'output_ch2.csv', 'output_ch3.csv', 'output_ch4.csv']

# 検証対象（生成させるコード）のファイル
TEST_FILE = "rdr_data.csv"

# 検証するShot数のリスト（0=Zero-Shot, 1, 2, 4, 8件の例示を与える）
SHOT_COUNTS_TO_TEST = [0, 1, 2, 4, 8]

# 検証データの件数（rdr_data.csvからランダムに抽出する数）
# 生成結果を目視確認する用途のため、10件程度が適切である
EVAL_SAMPLE_SIZE = 10

# 結果保存ファイル名
OUTPUT_FILE = "few_shot_generation_experiment_rdr.csv"

# --- 1. データ準備 ---
print("=== データ読み込み ===")

# (1) 例示用データ（Train Pool）の読み込み
train_df_list = []
for file_name in TRAIN_FILES:
    if os.path.exists(file_name):
        try:
            # BOM付きUTF-8に対応
            _df = pd.read_csv(file_name, encoding='utf-8-sig')
            train_df_list.append(_df)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

if not train_df_list:
    raise ValueError("例示用のCSVファイル（output_chX.csv）が見つかりません。")

train_pool_df = pd.concat(train_df_list, ignore_index=True)

# 必要なカラム（正解データ）がある行だけ残す
required_cols = ['original_code', 'branks_level1', 'branks_level2', 'branks_level3', 'branks_level4', 'branks_level5']
train_pool_df = train_pool_df.dropna(subset=required_cols)
print(f"Train Pool Size: {len(train_pool_df)} rows (例示候補)")

# (2) 検証用データ（Test Data）の読み込み
if os.path.exists(TEST_FILE):
    try:
        test_full_df = pd.read_csv(TEST_FILE, encoding='utf-8-sig')
        print(f"Loaded {TEST_FILE}: {len(test_full_df)} rows")
    except Exception as e:
        print(f"Error loading {TEST_FILE}: {e}")
        exit()
else:
    print(f"Error: {TEST_FILE} が見つかりません。")
    exit()

# 検証データの抽出
if len(test_full_df) > EVAL_SAMPLE_SIZE:
    test_df = test_full_df.sample(n=EVAL_SAMPLE_SIZE, random_state=42)
    print(f"Validation Target: {len(test_df)} items (Sampled from {TEST_FILE})")
else:
    test_df = test_full_df
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
    
    # Few-Shot (Train Poolからランダム抽出)
    if n_shots > 0 and len(train_pool) > 0:
        # データ数が足りない場合はあるだけ使う
        sample_count = min(len(train_pool), n_shots)
        shots = train_pool.sample(n=sample_count)
        
        for _, row in shots.iterrows():
            # 例示：(入力)元のコード -> (出力)正解の穴埋めコード
            messages.append({"role": "user", "content": f"{rule}\nコード:\n{row['original_code']}"})
            messages.append({"role": "assistant", "content": f"```nadesiko\n{row[target_col]}\n```"})
    
    # Target (Test Data)
    messages.append({"role": "user", "content": f"{rule}\nコード:\n{target_code}"})
    return messages

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

# --- 4. 生成実験実行 ---
results = []
print(f"\n=== 生成実験開始 (Shots: {SHOT_COUNTS_TO_TEST}) ===")
print("※自動評価（スコア算出）は行わず、生成結果の保存のみを行います。")

# Shot数のループ
for n_shots in SHOT_COUNTS_TO_TEST:
    print(f"\n>>> Running with n_shots = {n_shots} <<<")
    
    # レベルのループ
    for level in [1, 2, 3, 4, 5]:
        category_char = 'ABCDE'[level-1]
        
        # 検証データのループ
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"L{level}-Shot{n_shots}"):
            original_code = row['original_code']
            file_name = row.get('file_name', f'sample_{index}')
            
            # メッセージ作成 (Train Poolから例示を抽出)
            messages = create_chat_messages(original_code, level, train_pool_df, n_shots)
            
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
                
            except Exception as e:
                print(f"Error at Shot{n_shots}-L{level}-{index}: {e}")
                cleaned_text = "ERROR"

            results.append({
                "n_shots": n_shots,
                "level": level,
                "category": category_char,
                "file_name": file_name,
                "original_code": original_code,
                "generated_code": cleaned_text
            })

# --- 5. 結果保存 ---
results_df = pd.DataFrame(results)

# 見やすさのためソート
results_df = results_df.sort_values(by=['n_shots', 'level', 'file_name'])

results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\n実験完了。結果を {OUTPUT_FILE} に保存した。")
print("generated_code 列を確認し、例示数(n_shots)ごとの変化を分析できる。")