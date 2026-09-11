import pandas as pd
import torch
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-8B"
PEFT_MODEL_ID = "lorenzocazzador/coder-grpo-qwen3-8b"

# 入力ファイルと出力ファイルの設定
TARGET_FILES = ["/mnt/data/s26g375/project/add_python_result_qwen3.csv"]
OUTPUT_FILE = "structured_problems_qwen3.csv"

print("=== モデルおよびPEFTアダプターのロード中... ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# トークナイザーの読み込み
try:
    tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL_ID, trust_remote_code=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

# ベースモデルの読み込み
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# PEFTアダプターの統合
print(f"=== PEFTアダプター {PEFT_MODEL_ID} を統合中 ===")
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)

def apply_chat_template_safe(messages, tokenizer, add_generation_prompt=True):
    """Qwen3の思考モードを明示的にOFFにする。テンプレートが未対応の場合はフォールバック。"""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

def strip_thinking(text):
    """<think>ブロックを除去。閉じタグがない(=生成が思考の途中で打ち切られた)場合は
    有効な出力が得られていないとみなし空文字を返す。"""
    if "<think>" in text and "</think>" not in text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def generate_structured_problem(problem_text, tokenizer, model):
    """Qwen+PEFTモデルを使用して問題文の論理構造をJSONフォーマットで抽出する"""
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
    
    formatted_prompt = apply_chat_template_safe(messages, tokenizer)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        raw_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 思考ブロックの除去（未完了なら空文字になる＝JSON抽出をスキップ）
        stripped_text = strip_thinking(raw_text)
        if not stripped_text:
            return "{}"

        # JSON部分（{}で囲まれた範囲）を抽出
        match = re.search(r'\{.*\}', stripped_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
        else:
            clean_text = stripped_text.replace("```json", "").replace("```", "").strip()
        
        return clean_text
        
    except RuntimeError as e:
        if "out of memory" in str(e) or "CUDA" in str(e):
            print(f"\nError: GPU Memory overflow. Skipping.")
        return "{}"
    finally:
        del inputs
        if 'output_ids' in locals():
            del output_ids
        torch.cuda.empty_cache()

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
    raise ValueError("読み込めるCSVファイルがありません。パスを確認してください。")

full_df = pd.concat(df_list, ignore_index=True)
required_cols = ['original_code', 'generated_problem']
full_df = full_df.dropna(subset=required_cols)

print("\n=== 問題文の構造化（JSON抽出）開始 ===")
structured_texts = []
for idx, row in tqdm(full_df.iterrows(), total=len(full_df)):
    s_text = generate_structured_problem(row['generated_problem'], tokenizer, model)
    structured_texts.append(s_text)

full_df['structured_problem'] = structured_texts

empty_count = sum(1 for t in structured_texts if t == "{}")
if empty_count:
    print(f"注意: {empty_count} 件で有効なJSONが得られませんでした（思考未完了またはエラー）。")

print("\n=== 結果の保存 ===")
full_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"構造化されたデータを {OUTPUT_FILE} に保存しました。")