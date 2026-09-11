import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pandas as pd

# --- 1. セットアップ ---
model_id = "elyza/Llama-3-ELYZA-JP-8B"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# プロンプトの準備（コード生成用）
# システムプロンプトで「なでしこ3」の文法に従うよう強く指示する
gen_instructs = """
あなたは、日本語プログラミング言語「なでしこ3（nadesiko3）」の熟練プログラマである。
入力される「問題文」を詳細に分析し、その仕様を完全に満たす実行可能なコードを生成せよ。

### 制約事項：
1. 出力は、なでしこ3の実行可能なソースコードのみとする。
2. 解説や補足説明、コードブロックの記号（```）などは一切含めず、純粋なプログラムのテキストのみを出力せよ。
3. なでしこ3の標準的な文法（「～を表示」「～を～まで繰り返す」など）を使用せよ。
"""

# --- 2. 前段階で生成した結果（result.csv）の読み込み ---

input_csv = "/home/s22t324/project/result.csv"

try:
    df_result = pd.read_csv(input_csv)
    # 期待される列：filename, original_code, generated_problem
except Exception as e:
    print(f"CSVの読み込み中にエラーが発生した： {e}")
    exit()

eval_results = []

print(f"Processing {len(df_result)} problems for code generation...")

# 各問題文からコードを逆生成する
for index, row in df_result.iterrows():
    filename = row['filename']
    problem_text = row['generated_problem']
    
    print(f"--- Generating Code for ({index+1}/{len(df_result)}): {filename} ---")
    
    try:
        # チャットテンプレート
        chats = [
            { "role": "system", "content": gen_instructs},
            { "role": "user", "content": f"以下の問題文に従って、なでしこ3のコードを作成せよ。\n\n問題文：\n{problem_text}"},
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            chats,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded_input = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            return_attention_mask=True
        )

        with torch.no_grad():
            output_ids = model.generate(
                encoded_input["input_ids"].to(model.device),
                attention_mask=encoded_input["attention_mask"].to(model.device),
                do_sample=True,
                temperature=0.3, # 生成の安定性を高めるため少し低めに設定
                top_p=0.9,
                max_new_tokens=2048,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 生成されたコードをデコード
        generated_code = tokenizer.decode(output_ids.tolist()[0][encoded_input["input_ids"].size(1) :], skip_special_tokens=True)
        
        # 検証用に、元のコード、問題文、再生成コードをセットで保存する
        eval_results.append({
            'filename': filename,
            'original_code': row['original_code'],
            'generated_problem': problem_text,
            'reconstructed_code': generated_code.strip()
        })

    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")

# --- 3. 検証用データの保存 ---

if eval_results:
    df_eval = pd.DataFrame(eval_results)
    # 文字化け対策としてutf-8-sigで保存
    df_eval.to_csv("eval_reconstruction.csv", index=False, encoding='utf-8-sig')
    print("\n✅ All code generation tasks completed. Results saved to eval_reconstruction.csv")
else:
    print("\n⚠️ No output generated.")