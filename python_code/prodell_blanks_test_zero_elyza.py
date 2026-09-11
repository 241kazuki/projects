import pandas as pd
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 設定 ---
MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"
INPUT_FILE = "rdr_data.csv"     # 生成元のデータファイル
OUTPUT_FILE = "produire_generation_sample.csv" # 結果保存ファイル
SAMPLE_COUNT = 5                # 検証のために抜き出すコードの数

# --- 1. データ準備 ---
print("=== データ読み込み ===")
if os.path.exists(INPUT_FILE):
    try:
        # BOM付きUTF-8に対応
        full_df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
        print(f"Loaded {INPUT_FILE}: {len(full_df)} rows")
    except Exception as e:
        print(f"Error loading {INPUT_FILE}: {e}")
        exit()
else:
    print(f"Error: {INPUT_FILE} が見つかりません。")
    print("先にデータ作成コードを実行して rdr_data.csv を作成してください。")
    exit()

# 必要なカラムの確認
if 'original_code' not in full_df.columns:
    print("Error: CSVに 'original_code' カラムがありません。")
    exit()

# ランダムにサンプリング（データ数が指定数より少ない場合は全件使用）
actual_sample_count = min(len(full_df), SAMPLE_COUNT)
sample_df = full_df.sample(n=actual_sample_count, random_state=42)
print(f"検証用に {actual_sample_count} 件のコードをランダムに抽出しました。")

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

# --- 3. プロンプト作成関数（プロデル用） ---
def create_chat_messages(target_code, level):
    """
    プロデル（Produire）の文法に基づいたルール定義（Zero-Shot）
    """
    rules_text = {
        1: """【タスク: カテゴリA（データ・演算）に基づく空欄化】
以下の要素を「---」に置換してください。それ以外は変更しないでください。
1. 値: 数値、文字列の中身（「〜」）
2. 演算子: ＋ － × ÷ ＝ ＜ ＞ ≠ など""",
        
        2: """【タスク: カテゴリB（基本命令）に基づく空欄化】
以下のプロデル基本命令を「---」に置換してください。それ以外は変更しないでください。
1. 入出力: 表示する、報告する、尋ねる、待機する
2. 変数操作: 変数、(変数への)代入""",
        
        3: """【タスク: カテゴリC（構造・ロジック）に基づく空欄化】
以下の制御構文に関わる語句を「---」に置換してください。それ以外は変更しないでください。
1. 繰り返し: 繰り返す、回、間、抜ける、続ける
2. 条件分岐: もし、なら、そうでなければ、そして""",
        
        4: """【タスク: カテゴリD（応用・データ操作）に基づく空欄化】
以下の応用的なデータ操作語句を「---」に置換してください。それ以外は変更しないでください。
1. データ構造: 配列、一覧、辞書、要素、追加する
2. 手順定義: 手順、戻る、返す
3. その他: ファイル、読み込む、保存する""",
        
        5: """【タスク: カテゴリE（ドメイン固有）に基づく空欄化】
以下のGUI・オブジェクト指向的な命令を「---」に置換してください。それ以外は変更しないでください。
1. GUI部品: ウィンドウ、ボタン、ラベル、テキストボックス
2. イベント: クリックされた時、作られた時
3. 属性操作: 内容、位置、大きさ、閉じる"""
    }
    
    rule = rules_text.get(level, "コードの一部をルールに基づいて隠蔽してください。")
    
    messages = []
    messages.append({"role": "user", "content": f"{rule}\nコード:\n{target_code}"})
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
print(f"\n=== 生成開始 ({actual_sample_count}件 × 5レベル) ===")

# サンプリングした各コードに対して処理
for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    original_code = row['original_code']
    file_name = row.get('file_name', f'sample_{index}') # ファイル名があれば使用
    
    # レベル1〜5すべてを実行
    for level in [1, 2, 3, 4, 5]:
        category_char = 'ABCDE'[level-1]
        
        # プロンプト作成 & 推論
        messages = create_chat_messages(original_code, level)
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
            "original_code": original_code,
            "generated_code": cleaned_text
        })

# --- 5. 結果保存 ---
results_df = pd.DataFrame(results)

# 見やすさのためにソート（ファイル名順、レベル順）
results_df = results_df.sort_values(by=['file_name', 'level'])

results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n生成完了。結果を {OUTPUT_FILE} に保存しました。")
print("各ファイルの 'generated_code' 列を目視して、指定カテゴリ通りに空欄化されているか確認してください。")