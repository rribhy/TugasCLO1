import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from surprise.model_selection import KFold
from surprise import Dataset, Reader, accuracy, KNNBasic
from collections import defaultdict

path = "datasets6m/"
books = pd.read_csv(path + 'books.csv', sep=",")
ratings = pd.read_csv(path + 'ratings.csv', sep=",")
book_tags = pd.read_csv(path + 'book_tags.csv', sep=",")
tags = pd.read_csv(path + 'tags.csv', sep=",")

print("\n--- Info Ratings ---")
ratings.info()
print("\n--- Head Ratings ---")
print(ratings.head())

print("\n--- Info Books ---")
books.info()
print("\n--- Head Books ---")
print(books.head())

print("\nMembersihkan data...")

initial_rows = len(ratings)
duplicate_rows = ratings.duplicated(subset=['user_id', 'book_id']).sum()
print(f"Jumlah rating duplikat: {duplicate_rows}")
ratings = ratings.drop_duplicates(subset=['user_id', 'book_id'], keep='first')
print(f"Baris setelah menghapus duplikat: {len(ratings)}")

plt.figure(figsize=(10, 6))
ax = sns.countplot(x='rating', data=ratings)
ax.set_title('Distribusi Rating Buku')
ax.set_xlabel('Rating buku')
ax.set_ylabel('Jumlah rating')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
user_counts = ratings['user_id'].value_counts() 
sns.histplot(user_counts, bins=50, kde=False, color='cadetblue')
plt.title('Distribusi Jumlah Rating per Pengguna')
plt.xlabel('Jumlah Rating')
plt.ylabel('Jumlah Pengguna')
plt.show()

mean_user_ratings = ratings.groupby('user_id')['rating'].mean()
plt.figure(figsize=(10, 6))
sns.histplot(mean_user_ratings, bins=30, color='cadetblue')
plt.title('Distribusi Rata-rata Rating Pengguna')
plt.xlabel('Rata-rata Rating')
plt.ylabel('Jumlah Pengguna')
plt.show()

def ndcg_at_k(recommended_items, true_items_with_relevance, k=10):
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG) at k.
    """
    top_k_recs = recommended_items[:k]
    
    dcg = 0.0
    for i, item_id in enumerate(top_k_recs):
        if item_id in true_items_with_relevance:
            relevance = true_items_with_relevance[item_id]
            dcg += relevance / np.log2(i + 2)
            
    ideal_relevances = sorted(true_items_with_relevance.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances[:k]):
        idcg += relevance / np.log2(i + 2)
        
    if idcg == 0:
        return 0.0
        
    return dcg / idcg

user_counts = ratings['user_id'].value_counts()
active_users = user_counts[user_counts >= 10].index[:10000]
ratings_small = ratings[ratings['user_id'].isin(active_users)].copy()

# 2. (Opsional) Batasi juga jumlah buku, misal 3000 buku terpopuler
book_counts = ratings_small['book_id'].value_counts()
popular_books = book_counts.index[:3000]
ratings_small = ratings_small[ratings_small['book_id'].isin(popular_books)]

print("Jumlah user unik:", ratings_small['user_id'].nunique())
print("Jumlah buku unik:", ratings_small['book_id'].nunique())
print("Jumlah baris rating:", len(ratings_small))

# 3. Pakai ratings_small untuk Surprise, bukan ratings penuh
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    ratings_small[['user_id', 'book_id', 'rating']],
    reader
)

print("\nUser-based Collaborative Filtering KNN")

rmse_scores = []
mae_scores = []
mse_scores = []
nmse_scores = []
ndcg_scores = []

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

relevance_threshold = 4.0

algo = KNNBasic(sim_options={'name': 'cosine','user_based': True})
kf = KFold(n_splits=10)
k_ndcg = 10
print("\nMengevaluasi algoritma KNN dengan 10-fold cross-validation manual.")

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)

    # --- Metrik regresi ---
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    mse = accuracy.mse(predictions, verbose=False)

    true_ratings_np = np.array([pred.r_ui for pred in predictions])
    variance_of_ratings = np.var(true_ratings_np)
    nmse = mse / variance_of_ratings if variance_of_ratings > 0 else 0

    # --- Metrik klasifikasi (accuracy, precision, recall, F1) ---
    # y_true: apakah rating aktual relevan
    # y_pred: apakah rating prediksi relevan
    y_true = np.array([1 if pred.r_ui >= relevance_threshold else 0 for pred in predictions])
    y_pred = np.array([1 if pred.est >= relevance_threshold else 0 for pred in predictions])

    fold_accuracy = accuracy_score(y_true, y_pred)
    fold_precision = precision_score(y_true, y_pred, zero_division=0)
    fold_recall = recall_score(y_true, y_pred, zero_division=0)
    fold_f1 = f1_score(y_true, y_pred, zero_division=0)

    accuracy_scores.append(fold_accuracy)
    precision_scores.append(fold_precision)
    recall_scores.append(fold_recall)
    f1_scores.append(fold_f1)

    # --- NDCG seperti sebelumnya ---
    true_items_by_user = defaultdict(dict)
    recs_by_user = defaultdict(list)
    for pred in predictions:
        true_items_by_user[pred.uid][pred.iid] = pred.r_ui
        recs_by_user[pred.uid].append((pred.est, pred.iid))

    user_ndcgs = []
    for uid, true_items in true_items_by_user.items():
        sorted_recs = sorted(recs_by_user[uid], key=lambda x: x[0], reverse=True)
        recommended_item_ids = [iid for est, iid in sorted_recs]
        ndcg_val = ndcg_at_k(recommended_item_ids, true_items, k=k_ndcg)
        user_ndcgs.append(ndcg_val)

    fold_ndcg = np.mean(user_ndcgs) if user_ndcgs else 0

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mse_scores.append(mse)
    nmse_scores.append(nmse)
    ndcg_scores.append(fold_ndcg)

# Rata-rata semua metrik
rmse_hasil = np.mean(rmse_scores)
mae_hasil = np.mean(mae_scores)
mse_hasil = np.mean(mse_scores)
nmse_hasil = np.mean(nmse_scores)
ndcg_hasil = np.mean(ndcg_scores)

accuracy_hasil = np.mean(accuracy_scores)
precision_hasil = np.mean(precision_scores)
recall_hasil = np.mean(recall_scores)
f1_hasil = np.mean(f1_scores)

print("\n--- Hasil Evaluasi Rata-rata ---")
print(f"RMSE rata-rata: {rmse_hasil:.4f}")
print(f"MAE rata-rata: {mae_hasil:.4f}")
print(f"MSE rata-rata: {mse_hasil:.4f}")
print(f"NMSE rata-rata: {nmse_hasil:.4f}")
print(f"NDCG@{k_ndcg} rata-rata: {ndcg_hasil:.4f}")
print(f"Accuracy rata-rata: {accuracy_hasil:.4f}")
print(f"Precision rata-rata: {precision_hasil:.4f}")
print(f"Recall rata-rata: {recall_hasil:.4f}")
print(f"F1-score rata-rata: {f1_hasil:.4f}")

metrics = [
    'RMSE', 'MAE', 'MSE', 'NMSE',
    f'NDCG@{k_ndcg}', 'Accuracy', 'Precision', 'Recall', 'F1'
]
values = [
    rmse_hasil, mae_hasil, mse_hasil, nmse_hasil,
    ndcg_hasil, accuracy_hasil, precision_hasil, recall_hasil, f1_hasil
]

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=metrics, y=values)
plt.title('Hasil Evaluasi Rata-rata Algoritma KNN User-Based')
plt.xlabel('Metrik Evaluasi')
plt.ylabel('Nilai Rata-rata')
plt.ylim(0, max(values) * 1.2)

for i, v in enumerate(values):
    ax.text(i, v + (max(values) * 0.02), f"{v:.4f}", ha='center', va='bottom', fontweight='bold')

plt.show()