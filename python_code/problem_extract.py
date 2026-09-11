import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 設定 ---
MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"
INPUT_CSV = "result.csv" # ★変更: 先ほど評価に使用した10件のファイルから直接読み込む
OUTPUT_CSV = "structured_test_result.csv" # 抽出結果の保存先

def generate_structured_problem(generated_problem, tokenizer, model):
    """問題文を特定の言語に依存しない形に構造化する"""
    sys_msg = (
        "あなたは優秀なシステムエンジニアである。"
        "以下のプログラミングの問題文から、特定のプログラミング言語に依存しない形で、"
        "論理構造を【目的】【入力】【制御構造】【処理手順】【出力】の5項目に分けて箇条書きで抽出せよ。"
        "【制御構造】の項目には、ループ（繰り返し）や条件分岐などを明記すること。使用しない場合は「なし」とせよ。"
        "解説や挨拶は一切不要である。構造化されたテキストのみを出力せよ。"
    )
    
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"問題文:\n{generated_problem}"}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # GPUメモリ解放
    del inputs
    del output_ids
    torch.cuda.empty_cache()
    
    return raw_text.strip()

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"エラー: {INPUT_CSV} が見つからない。")
        return

    print("=== データ読み込み ===")
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    
    # ★変更: generate_info_result.csv のカラム名に合わせて 'generated_problem' を対象とする
    if 'generated_problem' not in df.columns:
        print("エラー: 'generated_problem' カラムが存在しない。")
        return
        
    test_df = df.dropna(subset=['generated_problem']).reset_index(drop=True)
    
    print("\n=== モデルロード中... ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    
    results = []
    print(f"\n=== 構造化抽出テスト開始 ({len(test_df)}件) ===")
    
    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        original_problem = row['generated_problem'] # ★変更
        
        # 構造化の実行
        structured_text = generate_structured_problem(original_problem, tokenizer, model)
        
        results.append({
            "original_problem": original_problem,
            "structured_problem": structured_text
        })
        
        # コンソールにも出力して即座に確認できるようにする
        print("\n" + "="*40)
        print(f"【元の問題文】\n{original_problem}")
        print("-" * 40)
        print(f"【抽出された構造化テキスト】\n{structured_text}")
        print("="*40 + "\n")

    # 結果をCSVに保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"抽出テスト完了。結果を {OUTPUT_CSV} に保存した。")

if __name__ == "__main__":
    main()