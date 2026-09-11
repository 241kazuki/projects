import pandas as pd
import re
from difflib import SequenceMatcher

# --- 設定 ---
# 前段階で作成した、比較対象のデータが含まれるファイル名
INPUT_FILE = "/home/s22t324/project/eval_reconstruction.csv"
OUTPUT_FILE = "quantitative_analysis_result.csv"

# --- 1. ユーティリティ関数 ---

def normalize_code(text):
    """
    比較のためにコードを正規化する。
    - 文字列化（NaN対策）
    - 空白、タブ、改行の削除
    - 全角スペースの削除
    - 読点（、）や句点（。）の扱いはなでしこ特有だが、
      構造比較のためには一旦すべて除去して「純粋なトークン順序」に近づける。
    """
    if pd.isna(text):
        return ""
    # 空白文字全般を削除
    text = re.sub(r'\s+', '', str(text))
    # なでしこ特有の記号（句読点）を削除（構造一致の判定を甘くする場合）
    text = text.replace('、', '').replace('。', '')
    return text

def calculate_scores(original, reconstructed):
    """
    正解類似度と構造一致度を計算する。
    """
    # 文字列としての類似度（SequenceMatcher）
    # 元の改行や空白を維持した状態で比較
    similarity = SequenceMatcher(None, str(original), str(reconstructed)).ratio()

    # 構造的一致度（正規化後の比較）
    orig_norm = normalize_code(original)
    reco_norm = normalize_code(reconstructed)
    
    # 正規化した文字列同士での完全一致を 1.0, 不一致を 0.0 とする
    structure_match = 1.0 if orig_norm == reco_norm else 0.0
    
    # 正規化後の類似度も参考値として計算
    norm_similarity = SequenceMatcher(None, orig_norm, reco_norm).ratio()

    return similarity, structure_match, norm_similarity

# --- 2. 分析の実行 ---

print(f"=== 分析開始: {INPUT_FILE} ===")

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"エラー: {INPUT_FILE} が見つかりません。")
    exit()

analysis_results = []

for index, row in df.iterrows():
    orig = row['original_code']
    reco = row['reconstructed_code']
    filename = row['filename']

    sim, struct, norm_sim = calculate_scores(orig, reco)
    
    analysis_results.append({
        'filename': filename,
        'similarity_score': sim,           # 原文の類似度
        'structure_match': struct,         # 構造の完全一致 (0 or 1)
        'normalized_similarity': norm_sim  # 構造の類似度
    })

# 結果を結合
df_analysis = pd.DataFrame(analysis_results)
df_final = pd.concat([df, df_analysis.drop('filename', axis=1)], axis=1)

# --- 3. 集計と保存 ---

print("\n=== 分析結果サマリ ===")
summary = {
    '平均正解類似度': df_final['similarity_score'].mean(),
    '構造完全一致率': df_final['structure_match'].mean(),
    '平均構造類似度': df_final['normalized_similarity'].mean()
}

for key, val in summary.items():
    print(f"{key}: {val:.4f}")

# CSVに保存
df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n✅ 分析が完了した。結果は {OUTPUT_FILE} に保存された。")