import pandas as pd
import torch
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 設定 ---
# ELYZAモデルを指定
MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"

# 入力ファイルと出力ファイルの設定
TARGET_FILES = ["/home/s22t324/project/add_python_result.csv"]
OUTPUT_FILE = "structured_problems_elyza.csv"

# --- 1. モデル準備 ---
print("=== モデルロード中... ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

# --- 2. ユーティリティ関数 ---
def generate_structured_problem(problem_text, tokenizer, model):
    """ELYZAを使用して問題文の論理構造をJSONフォーマットで抽出する"""
    sys_msg = (
        "あなたは優秀なシステムエンジニアである。"
        "以下のプログラミングの問題文から、特定のプログラミング言語に依存しない形で、"
        "アルゴリズムの骨格となる論理構造を詳細に抽出し、以下のJSONフォーマットで出力せよ。\n\n"
        "{\n"
        "  \"目的\": \"プログラムの全体的な目的\",\n"
        "  \"入力データ\": \"外部からの入力や初期化される変数（例：1から6の乱数、空の配列など具体的に）\",\n"
        "  \"制御構造\": [\"無限ループ\", \"条件分岐（AとBが一致する場合）\", \"関数定義\" など、具体的な条件を含めたロジックのリスト。ない場合は空配列],\n"
        "  \"処理手順\": [\n"
        "    \"1. 〇〇を初期化する\",\n"
        "    \"2. 無限ループを開始する\",\n"
        "    \"3. 条件（X>0）の場合は〇〇を実行する\"\n"
        "  ],\n"
        "  \"出力データ\": \"画面描画、文字列表示などの出力\",\n"
        "  \"制約事項\": \"特定のエラー処理や制限\"\n"
        "}\n\n"
        "注意：\n"
        "- 処理手順は単なる自然言語の要約ではなく、ブロック構造（ループの開始・終了、条件分岐のトリガー）が明確にわかるように細かく抽出すること。\n"
        "- PythonやJavaなどの特定のプログラミング言語の構文（例：while True, if, ==）は絶対に含めないこと。\n"
        "- 解説やマークダウン記法（```json など）は一切不要である。純粋なJSONテキストのみを出力せよ。"
    )
    
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"問題文:\n{problem_text}"}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    try:
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1024, # 詳細な構造化データを出力しきれるよう上限を拡張
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        raw_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # インポート済みのreモジュールを使用し、JSON部分（{}で囲まれた範囲）のみを確実に抽出する
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
        else:
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        return clean_text
        
    except RuntimeError as e:
        if "out of memory" in str(e) or "CUDA" in str(e):
            print(f"\nError: GPU Memory overflow. Skipping.")
        return "{}"
    finally:
        # OOM(メモリ不足)対策のため都度メモリを解放
        if 'inputs' in locals():
            del inputs
        if 'output_ids' in locals():
            del output_ids
        torch.cuda.empty_cache()

# --- 3. メイン処理 ---
print("\n=== データ読み込み ===")
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
    raise ValueError("読み込めるCSVファイルがない。パスを確認すること。")

# 全データを統合し、必要なカラムが欠損している行を削除
full_df = pd.concat(df_list, ignore_index=True)
required_cols = ['original_code', 'generated_problem']
full_df = full_df.dropna(subset=required_cols)

print("\n=== 問題文の構造化（JSON抽出）開始 ===")
structured_texts = []
# 進捗状況を表示しながら全データの構造化を実行
for idx, row in tqdm(full_df.iterrows(), total=len(full_df)):
    s_text = generate_structured_problem(row['generated_problem'], tokenizer, model)
    structured_texts.append(s_text)

# 抽出した論理構造を新しいカラムとしてデータフレームに追加
full_df['structured_problem'] = structured_texts

# 結果をCSVファイルに保存
print("\n=== 結果の保存 ===")
full_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"構造化されたデータを {OUTPUT_FILE} に保存した。")