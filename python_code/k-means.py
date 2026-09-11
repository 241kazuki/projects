"""
check_cluster_distribution.py

元のスクリプトと同じデータ読み込み・K-Meansクラスタリング処理を再現し、
「各クラスタに元データが何件ずつ含まれているか」だけを出力する。

元のスクリプトが省略していた full_df['cluster'].value_counts() 相当の
情報を得るためのものです。モデルのロードや生成は一切行わないため、
GPU無しでもすぐに実行できます。
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --- 設定（元のスクリプトと同じ） ---
TARGET_FILES = ["structured_problems_qwen3.csv", "structured_problems.csv"]
NUM_CLUSTERS = 10
RANDOM_STATE = 123

print("=== データ読み込みと統合 ===")
df_list = []
for file_name in TARGET_FILES:
    if os.path.exists(file_name):
        try:
            _df = pd.read_csv(file_name, encoding="utf-8-sig")
            print(f"Loaded {file_name}: {_df.shape[0]} rows")
            df_list.append(_df)
            break  # 元のスクリプトと同じく、最初に見つかったファイルのみ使用
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    else:
        print(f"Warning: {file_name} not found.")

if not df_list:
    raise ValueError("読み込めるCSVファイルがありません。パスを確認してください。")

full_df = pd.concat(df_list, ignore_index=True)
required_cols = ["original_code", "generated_problem", "structured_problem"]
full_df = full_df.dropna(subset=required_cols).reset_index(drop=True)

print(f"\n有効データ件数: {len(full_df)} 件")

if len(full_df) <= NUM_CLUSTERS:
    print("データ件数がクラスタ数以下のため、クラスタリングは行われません（元スクリプトも同様）。")
else:
    print(f"\n=== K-Means クラスタリング（K={NUM_CLUSTERS}） ===")
    texts = full_df["original_code"].fillna("").tolist()

    # 元のスクリプトと全く同じベクトル化・クラスタリング設定
    cluster_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    X = cluster_vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    full_df["cluster"] = kmeans.fit_predict(X)

    # --- ここが元スクリプトになかった集計処理 ---
    counts = full_df["cluster"].value_counts().sort_index()
    percentages = (counts / len(full_df) * 100).round(1)

    summary = pd.DataFrame({
        "cluster": counts.index,
        "count": counts.values,
        "percentage(%)": percentages.values,
    })

    print("\n=== クラスタ別 件数・比率 ===")
    print(summary.to_string(index=False))

    print(f"\n合計件数: {summary['count'].sum()} 件")
    print(f"最大クラスタ: {summary['count'].max()} 件 "
          f"（cluster {summary.loc[summary['count'].idxmax(), 'cluster']}）")
    print(f"最小クラスタ: {summary['count'].min()} 件 "
          f"（cluster {summary.loc[summary['count'].idxmin(), 'cluster']}）")

    # 参考: どのクラスタが評価セット(10件)として抽出されたか、実行のたびに再現したい場合は
    # 元スクリプトと同じ random_state=123 で1件ずつサンプリングすれば同じ10件が得られます。

    out_path = "cluster_distribution.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n詳細を {out_path} に保存しました。")