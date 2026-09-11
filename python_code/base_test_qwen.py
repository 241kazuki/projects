import pandas as pd
import torch
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-8B"
PEFT_MODEL_ID = "lorenzocazzador/coder-grpo-qwen3-8b"
INPUT_FILE = "result.csv"                        # 読み込む入力ファイル
OUTPUT_FILE = "base_test_result_qwen3_8b_grpo.csv" # 最終結果の保存先
MAX_INPUT_TOKENS = 4000                          # Few-shotプロンプトの最大トークン数目安

def load_model():
    print(f"=== ベースモデル {BASE_MODEL_ID} をロード中 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # トークナイザーのロード (PEFTリポジトリ優先、無ければベースモデルから)
    try:
        tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL_ID, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"=== PEFTアダプター {PEFT_MODEL_ID} を統合中 ===")
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
    return tokenizer, model

def call_qwen(messages, tokenizer, model, max_new_tokens=1024):
    """Qwen+PEFTモデルでテキストを生成する共通関数"""
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
                max_new_tokens=max_new_tokens,
                temperature=0.2, # コード生成の正確性を高めるため低めに設定
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # 入力プロンプト部分をカットして生成部分のみを取得
        output_text = tokenizer.decode(output_ids.tolist()[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        return output_text.strip()
    finally:
        # GPUメモリの解放
        del inputs
        if 'output_ids' in locals():
            del output_ids
        torch.cuda.empty_cache()

def generate_problem(code, tokenizer, model):
    system_msg = (
        "あなたはプログラミング教材の作成者である。\n"
        "入力された日本語プログラミング言語「なでしこ」のコードの動作が完全に再現できるような、"
        "詳細な仕様が含まれた自然で論理的な問題文を生成せよ。\n"
        "【制約】箇条書きやマークダウンは使わず、1つの連続した文章にすること。"
        "Pythonなどの他言語の専門用語(while, import, 辞書など)は避け、自然な日本語で説明すること。"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"以下のコードから問題文を作成せよ:\n\n{code}"}
    ]
    return call_qwen(messages, tokenizer, model, max_new_tokens=1024)

def extract_code_block(text):
    """生成されたテキストからコードブロックのみを抽出する"""
    pattern = r"```(?:nadesiko)?\s*(.*?)(?:```|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1)
        if "```" in content:
            content = content.split("```")[0]
        return content.strip()
    return re.sub(r'###.*', '', text).strip()

def normalize_text(text):
    """比較のために空白や改行を除去する"""
    text_no_hyphen = re.sub(r'-+', '', str(text))
    return "".join(text_no_hyphen.split())

def count_tokens(messages, tokenizer):
    """メッセージリストの総トークン数を概算する"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))

def create_few_shot_messages(problem_text, train_pool, tokenizer, max_tokens):
    rule = (
        "以下の問題文の指示に従って、プログラミング言語「なでしこ」のコードを出力すること。\n"
        "【なでしこ記述時の絶対ルール】\n"
        "1. 助詞の重要性: 「て」「に」「を」「は」などの助詞が関数の引数を決定する極めて重要な役割を持つ。助詞は省略せず正確に記述すること。\n"
        "2. 語順: 日本語の自然な語順に従うこと（例: AにBを足す）。\n"
        "3. 制御構造: 「もし〜ならば」「〜の間」「〜回」などを正しく用いること。\n"
        "解説は一切不要である。コードのみを出力すること。"
    )
    base_messages = [{"role": "user", "content": f"{rule}\n問題:\n{problem_text}"}]
    shots_messages = []
    
    for _, row in train_pool.iterrows():
        new_shot = [
            {"role": "user", "content": f"{rule}\n問題:\n{row['generated_problem']}"},
            {"role": "assistant", "content": f"```nadesiko\n{row['original_code']}\n```"}
        ]
        potential_shots = shots_messages + new_shot
        
        # トークン数が上限を超えない範囲でFew-shotを追加
        if count_tokens(potential_shots + base_messages, tokenizer) < max_tokens:
            shots_messages = potential_shots
        else:
            break
            
    return shots_messages + base_messages

def main():
    print(f"=== ファイル {INPUT_FILE} を読み込みます ===")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        
    if 'original_code' not in df.columns:
        raise ValueError(f"エラー: {INPUT_FILE} に 'original_code' カラムが存在しません。")

    # 空データの除去
    df = df.dropna(subset=['original_code']).reset_index(drop=True)
    print(f"対象データ: {len(df)} 件")

    tokenizer, model = load_model()

    # ==========================================
    # フェーズ1: original_code から 問題文 を生成
    # ==========================================
    print(f"\n=== フェーズ1: コードから問題文を生成 (Base: {BASE_MODEL_ID}, PEFT: {PEFT_MODEL_ID}) ===")
    generated_problems = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="問題文生成"):
        code = row['original_code']
        problem = generate_problem(code, tokenizer, model)
        generated_problems.append(problem)
    
    df['generated_problem'] = generated_problems

    # ==========================================
    # フェーズ2: 生成した問題文から コード を復元・評価
    # ==========================================
    print("\n=== フェーズ2: 問題文からコードを生成・評価 ===")
    results = []
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    
    texts = df['generated_problem'].tolist()
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="コード生成・評価"):
        problem_text = row['generated_problem']
        correct_answer = row['original_code']
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        similar_indices = [i for i, score in sim_scores if i != idx]
        
        train_pool = df.iloc[similar_indices].copy()
        
        messages = create_few_shot_messages(problem_text, train_pool, tokenizer, MAX_INPUT_TOKENS)
        
        raw_output = call_qwen(messages, tokenizer, model, max_new_tokens=1024)
        cleaned_code = extract_code_block(raw_output)
        
        corr_norm = normalize_text(correct_answer)
        gen_norm = normalize_text(cleaned_code)
        similarity_score = SequenceMatcher(None, corr_norm, gen_norm).ratio()
        
        results.append({
            "original_code": correct_answer,
            "generated_problem": problem_text,
            "generated_code": cleaned_code,
            "similarity": similarity_score
        })

    result_df = pd.DataFrame(results)
    print("\n=== 処理完了 ===")
    print(f"平均類似度スコア: {result_df['similarity'].mean():.4f}")
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"すべての処理が完了しました。結果は {OUTPUT_FILE} に保存されました。")

if __name__ == "__main__":
    main()