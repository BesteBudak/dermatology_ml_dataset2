import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 1️⃣ Veri setini oku
columns = [
    "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
    "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
    "knee_and_elbow_involvement", "scalp_involvement", "family_history",
    "melanin_incontinence", "eosinophils_infiltrate", "PNL_infiltrate",
    "fibrosis_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
    "parakeratosis", "clubbing_rete_ridges", "elongation_rete_ridges",
    "thinning_suprapapillary_epidermis", "spongiform_pustule",
    "munro_microabcess", "focal_hypergranulosis", "disappearance_granular_layer",
    "vacuolisation_damage_basal_layer", "spongiosis", "saw_tooth_retes",
    "follicular_horn_plug", "perifollicular_parakeratosis",
    "inflammatory_monoluclear_infiltrate", "band_like_infiltrate",
    "age", "class"
]

df = pd.read_csv("dermatology2.csv", header=None, names=columns, na_values='?')

# 2️⃣ Eksik değer doldurma
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["age"].fillna(df["age"].mean(), inplace=True)

# 3️⃣ Hedef ve öznitelikleri ayır 
# burda 6 hastalığa birden bakıyor. Yani senin hedef değişkenin (target) çok sınıflı bir sınıflandırma problemi.
X = df.drop("class", axis=1)
y = df["class"].astype(int)

# 4️⃣ Eğitim/test seti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5️⃣ Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Modeller
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42)
}

# 7️⃣ Modelleri eğit ve değerlendir
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} doğruluk: {acc:.4f}")
    results.append((name, acc))

# 8️⃣ En iyi model
best_model = max(results, key=lambda x: x[1])
print("\n✅ En iyi model:", best_model[0], "-> Doğruluk:", best_model[1])

# 9️⃣ Sınıflandırma raporu
best_clf = models[best_model[0]]
y_pred_best = best_clf.predict(X_test_scaled)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_best))

#ÖZNİTELİK AZALTILMIŞ HALİNİN KODLARI
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ------------------------
# # 1️⃣ Veri yükleme ve hazırlık
# # ------------------------
# columns = [
#     "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
#     "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
#     "knee_and_elbow_involvement", "scalp_involvement", "family_history",
#     "melanin_incontinence", "eosinophils_infiltrate", "PNL_infiltrate",
#     "fibrosis_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
#     "parakeratosis", "clubbing_rete_ridges", "elongation_rete_ridges",
#     "thinning_suprapapillary_epidermis", "spongiform_pustule",
#     "munro_microabcess", "focal_hypergranulosis", "disappearance_granular_layer",
#     "vacuolisation_damage_basal_layer", "spongiosis", "saw_tooth_retes",
#     "follicular_horn_plug", "perifollicular_parakeratosis",
#     "inflammatory_monoluclear_infiltrate", "band_like_infiltrate",
#     "age", "class"
# ]

# df = pd.read_csv("dermatology2.csv", header=None, names=columns, na_values='?')

# # Eksik yaş değerlerini doldur
# df["age"] = pd.to_numeric(df["age"], errors="coerce")
# df["age"].fillna(df["age"].mean(), inplace=True)

# # Hedef ve özellikler
# X = df.drop("class", axis=1)
# y = df["class"].astype(int)

# # Eğitim/Test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # ------------------------
# # 2️⃣ Verileri ölçekle
# # ------------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ------------------------
# # 3️⃣ Feature importance ile öznitelik seçimi (Random Forest)
# # ------------------------
# rf = RandomForestClassifier(n_estimators=200, random_state=42)
# rf.fit(X_train_scaled, y_train)

# importances = rf.feature_importances_
# feat_imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
# feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)

# # Önemli öznitelikleri görselleştir
# plt.figure(figsize=(12,6))
# sns.barplot(x='importance', y='feature', data=feat_imp_df)
# plt.title("Feature Importance - Random Forest")
# plt.tight_layout()
# plt.show()

# # Önemsiz öznitelikleri çıkar (importance < 0.02)
# threshold = 0.02
# selected_features = feat_imp_df[feat_imp_df['importance'] >= threshold]['feature'].tolist()
# print("\nSeçilen öznitelikler:", selected_features)

# X_train_sel = X_train[selected_features]
# X_test_sel = X_test[selected_features]

# X_train_scaled_sel = scaler.fit_transform(X_train_sel)
# X_test_scaled_sel = scaler.transform(X_test_sel)

# # ------------------------
# # 4️⃣ Modelleri tanımla (7 model)
# # ------------------------
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=500, multi_class='auto'),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
#     "KNN": KNeighborsClassifier(),
#     "SVC": SVC(),
#     "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
#     "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42)
# }

# # ------------------------
# # 5️⃣ Modelleri eğit ve değerlendir
# # ------------------------
# results = []
# for name, model in models.items():
#     model.fit(X_train_scaled_sel, y_train)
#     y_pred = model.predict(X_test_scaled_sel)
#     acc = accuracy_score(y_test, y_pred)
#     results.append((name, acc))
#     print(f"{name} doğruluk: {acc:.4f}")

# # En iyi modeli bul
# best_model = max(results, key=lambda x: x[1])
# print("\n✅ En iyi model:", best_model[0], "-> Doğruluk:", best_model[1])

# # Detaylı sınıflandırma raporu
# best_clf = models[best_model[0]]
# y_pred_best = best_clf.predict(X_test_scaled_sel)
# print("\nSınıflandırma Raporu:")
# print(classification_report(y_test, y_pred_best))
