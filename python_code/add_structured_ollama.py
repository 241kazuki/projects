import pandas as pd
import requests
import re
import os
from tqdm import tqdm

# --- 設定 ---
# OllamaのAPIエンドポイントとDeepSeek-Coderのモデル名を指定
OLLAMA_API_URL = "http://127.0.0.1:11460/api/chat"
OLLAMA_MODEL = "deepseek-coder-v2" # 起動しているモデル名に合わせて変更

# 入力ファイルと出力ファイルの設定
TARGET_FILES = ["/home/s22t324/project/add_python_result.csv"]
OUTPUT_FILE = "structured_problems.csv"

# --- 1. ユーティリティ関数 ---
def generate_structured_problem(problem_text):
    """DeepSeekを使用して問題文の論理構造をJSONフォーマットで抽出する"""
    sys_msg = (
        "あなたは優秀なシステムエンジニアである。"
        "以下のプログラミングの問題文から、特定のプログラミング言語に依存しない形で、"
        "アルゴリズムの論理構造を抽出し、以下のJSONフォーマットで出力せよ。\n\n"
        "{\n"
        "  \"目的\": \"プログラムの全体的な目的\",\n"
        "  \"入力データ\": \"外部からの入力や初期化される変数\",\n"
        "  \"制御構造\": [\"ループ\", \"条件分岐\", \"関数定義\" など必要なロジックのリスト。ない場合は空配列],\n"
        "  \"処理手順\": [\"手順1\", \"手順2\"...],\n"
        "  \"出力データ\": \"画面描画、文字列表示などの出力\",\n"
        "  \"制約事項\": \"特定のエラー処理や制限\"\n"
        "}\n\n"
        "解説やマークダウン記法（```json など）は一切不要である。純粋なJSONテキストのみを出力せよ。"
    )
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"問題文:\n{problem_text}"}
        ],
        "stream": False,
        "options": {"temperature": 0.0} # 構造化データの生成なのでランダム性をなくす
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code != 200:
            print(f"APIエラー詳細: {response.text}")
        response.raise_for_status()
        
        raw_text = response.json().get("message", {}).get("content", "")
        
        # DeepSeek-R1などの推論モデルを使用した場合に備え、思考プロセスを除去
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        
        # 不要なマークダウン記法を除去して純粋なテキスト(JSON)にする
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        return clean_text
        
    except Exception as e:
        print(f"Error during API call: {e}")
        return "{}" # エラー時は空のJSON風文字列を返す

# --- 2. メイン処理 ---
print("=== データ読み込み ===")
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
    s_text = generate_structured_problem(row['generated_problem'])
    structured_texts.append(s_text)

# 抽出した論理構造を新しいカラムとしてデータフレームに追加
full_df['structured_problem'] = structured_texts

# 結果をCSVファイルに保存
print("\n=== 結果の保存 ===")
full_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"構造化されたデータを {OUTPUT_FILE} に保存した。")