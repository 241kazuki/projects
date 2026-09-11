import requests
import pandas as pd
import re

# --- 1. セットアップ ---
# 変更したOllamaサーバーのポート番号に合わせて指定
OLLAMA_API_URL = "http://127.0.0.1:11460/api/chat"

# 使用するモデル名（"llama3.1", "deepseek-coder-v2", "deepseek-r1:7b" など）
MODEL_NAME = "deepseek-coder-v2"

# --- 2. AIへの指示 ---
instructs = """
あなたはコード変換器です。

入力として以下が与えられます。

1. 問題文
2. なでしこコード

あなたの仕事は、
問題文とコードの両方を参考にしながら、
入力されたなでしこコードと
完全に同じ動作をする
Python 3 のコードを生成することです。

## 厳守事項

- Pythonコードのみを出力してください
- 説明は禁止
- コメントは禁止
- markdown記法は禁止
- ``` や 
```python を出力しない
- 標準入力・標準出力を使用する
- 元コードと完全に同じ動作にする
- 問題文の仕様を必ず反映する
- 不明な場合でも可能な限り推測して実装する

## 出力
Pythonコードのみ
"""

# --- 3. CSV読み込み ---
input_csv = "/home/s22t324/project/result.csv"

try:
    # CSVに filename, generated_problem, original_code がある前提
    df_input = pd.read_csv(
        input_csv,
        dtype=str
    )
except Exception as e:
    print(f"CSV読み込みエラー: {e}")
    exit()

# --- 4. 結果保存用 ---
results_list = []

# --- 5. 各コードを処理 ---
for _, row in df_input.iterrows():
    filename = row["filename"]
    problem = row["generated_problem"]
    original_code = row["original_code"]

    try:
        # ユーザー入力作成
        user_content = f"""
# 問題文
{problem}

# なでしこコード
{original_code}
"""

        # Ollama API 用のペイロード構築
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": instructs},
                {"role": "user", "content": user_content}
            ],
            "stream": False,
            "options": {
                # 評価の再現性を保つため temperature を 0.0 に設定
                "temperature": 0.0
            }
        }

        # APIリクエスト
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # モデルの出力を取得
        output = result.get("message", {}).get("content", "")

        # DeepSeek-R1等の推論モデルが混ざった場合のフェイルセーフ（<think>タグの除去）
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)

        # markdown除去 (プロンプトで指示していても出力されるケースへの対策)
        output = output.replace("```python", "")
        output = output.replace("```", "")
        output = output.strip()

        # 空出力除外
        if output == "":
            print(f"{filename} の出力が空のためスキップします。")
            continue

        # 保存
        results_list.append({
            "filename": filename,
            "generated_problem": problem,
            "original_code": original_code,
            "python_code": output
        })
        
        print(f"処理完了: {filename}")

    except Exception as e:
        print(f"処理エラー ({filename}): {e}")
        continue

# --- 6. CSV保存 ---
if results_list:
    df_result = pd.DataFrame(results_list)
    output_csv = "add_python_result.csv"

    df_result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\n全ての処理が完了した。結果を {output_csv} に保存した。")
else:
    print("処理されたデータが存在しない。")