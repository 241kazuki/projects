import pandas as pd
import torch
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
# 例として /data/s22t324/hf_cache を指定していますが、ご自身の環境で空き容量のあるパスに変更してください
os.environ["HF_HOME"] = "/mnt/data/s26g375"
# --- 設定 ---
MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"
INPUT_FILE = "result.csv"  # 読み込む問題データセット

# ★ ここでFew-shotの例示数（ショット数）を変更できます
NUM_SHOTS = 3             

# --- プロンプト定義 ---
SYSTEM_RULE = (
    "あなたは日本語プログラミング言語「なでしこ」の熟練エンジニアである。\n"
    "入力された問題文の要件を満たす、文法破綻のないなでしこコードを生成せよ。\n\n"
    "【厳守すべき重要ルール】\n"
    "1. なでしこ固有のブロック構造（「〜の間繰り返す」「●〜とは」「もし〜ならば〜違えば」「ここまで」等）を正しく用いること。\n"
    "2. 助詞（「て」「に」「を」「は」等）は引数を決定するため省略せず正確に記述すること。\n"
    "3. 絶対に、Python等の他言語の構文（while, break, for, 括弧()など）を混入させないこと。\n"
    "4. 出力は純粋ななでしこコードのみとし、解説やMarkdown記法は一切不要である。"
)

def load_model():
    print(f"=== モデル {MODEL_ID} をロード中 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model

def generate_code(messages, tokenizer, model):
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        return_attention_mask=True
    ).to(model.device)
    
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            )
        output_text = tokenizer.decode(output_ids.tolist()[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        return extract_code_block(output_text)
    finally:
        del inputs
        if 'output_ids' in locals():
            del output_ids
        torch.cuda.empty_cache()

def extract_code_block(text):
    pattern = r"```(?:nadesiko)?\s*(.*?)(?:```|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1)
        if "```" in content:
            content = content.split("```")[0]
        return content.strip()
    return text.strip()

def main():
    # 1. データ読み込み
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        
    df = df.dropna(subset=['original_code', 'generated_problem']).reset_index(drop=True)
    
    # データ数がNUM_SHOTSより少ない場合の安全対策
    global NUM_SHOTS
    if len(df) - 1 < NUM_SHOTS:
        print(f"警告: データプールが少ないため、ショット数を {len(df) - 1} に制限します。")
        NUM_SHOTS = len(df) - 1
    
    # 2. 対象問題をランダムに1件選択
    target_idx = random.randint(0, len(df) - 1)
    target_row = df.iloc[1]
    problem_text = target_row['generated_problem']
    correct_code = target_row['original_code']
    
    print("\n" + "="*50)
    print("🎯 【ターゲット問題】")
    print("="*50)
    print(problem_text)
    print("\n✅ 【正解コード】")
    print(correct_code)
    print("="*50 + "\n")

    tokenizer, model = load_model()

    # ==========================================
    # 検証 1: Zero-shot (例示なし)
    # ==========================================
    print("\n🚀 [検証 1] Zero-shot で生成中...")
    zero_shot_messages = [
        {"role": "system", "content": SYSTEM_RULE},
        {"role": "user", "content": f"問題文:\n{problem_text}"}
    ]
    zero_shot_result = generate_code(zero_shot_messages, tokenizer, model)
    print("--- [Zero-shot 生成結果] ---")
    print(zero_shot_result)
    print("----------------------------")

    # ==========================================
    # 検証 2: Random Few-shot
    # ==========================================
    print(f"\n🚀 [検証 2] Random Few-shot ({NUM_SHOTS}件) で生成中...")
    # ターゲット以外から指定されたショット数だけランダムに抽出
    pool_indices = [i for i in range(len(df)) if i != target_idx]
    few_shot_indices = random.sample(pool_indices, NUM_SHOTS)
    
    few_shot_messages = [{"role": "system", "content": SYSTEM_RULE}]
    for idx in few_shot_indices:
        shot_row = df.iloc[idx]
        few_shot_messages.append({"role": "user", "content": f"問題文:\n{shot_row['generated_problem']}"})
        few_shot_messages.append({"role": "assistant", "content": f"```nadesiko\n{shot_row['original_code']}\n```"})
    
    # 最後にターゲット問題を追加
    few_shot_messages.append({"role": "user", "content": f"問題文:\n{problem_text}"})
    
    random_few_shot_result = generate_code(few_shot_messages, tokenizer, model)
    print("--- [Random Few-shot 生成結果] ---")
    print(random_few_shot_result)
    print("----------------------------------")
    
    print("\n💡 完了しました。結果を比較してスライド用の良い例を探してください。")

if __name__ == "__main__":
    main()