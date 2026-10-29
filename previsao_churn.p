"""
Desafio de Machine Learning: Previsão e Diagnóstico de Churn - TechGrow
Pipeline: EDA -> Pré-processamento -> Treino (LogReg, RF, XGBoost) -> Avaliação -> Insights
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# Tentativa de import do XGBoost; usa HistGradientBoostingClassifier como fallback nativo se não instalado
try:
    from xgboost import XGBClassifier
    tem_xgboost = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    tem_xgboost = False

# ==========================================================
# 1. IMPORTE E EXPLORE OS DADOS (EDA)
# ==========================================================
print("=" * 65)
print("1. CARREGAMENTO E ANÁLISE EXPLORATÓRIA DOS DADOS")
print("=" * 65)

nome_arquivo = "Caso_Pratico_base_churn_saas.csv"
caminho_dados = nome_arquivo if os.path.exists(nome_arquivo) else os.path.join("dados", nome_arquivo)
df = pd.read_csv(caminho_dados)

print(f"\nDimensões do dataset: {df.shape[0]} linhas e {df.shape[1]} colunas.")
print("\n--- Estrutura e tipos de dados ---")
print(df.dtypes)

print("\n--- Primeiras 5 linhas ---")
print(df.head())

print("\n--- Valores nulos por coluna ---")
print(df.isnull().sum())

print("\n--- Resumo estatístico ---")
print(df.describe())

# Distribuição da variável alvo (churn)
taxa_churn = df["churn"].mean() * 100
print(f"\nTaxa global de Churn: {taxa_churn:.2f}%")

# Gráfico da distribuição do Churn
plt.figure(figsize=(6, 4))
df["churn"].value_counts().plot(kind="bar", color=["#3498db", "#e74c3c"], edgecolor="black")
plt.title("Distribuição da Variável Alvo (0 = Manteve, 1 = Cancelou)")
plt.xlabel("Status de Churn")
plt.ylabel("Quantidade de Clientes")
plt.xticks(ticks=[0, 1], labels=["Ativo (0)", "Cancelou (1)"], rotation=0)
plt.tight_layout()
plt.savefig("distribuicao_churn.png")
plt.close()

# ==========================================================
# 2. TRATE E PREPARE OS DADOS
# ==========================================================
print("\n" + "=" * 65)
print("2. PREPARAÇÃO DOS DADOS, ENCODING E NORMALIZAÇÃO")
print("=" * 65)

# Remoção de identificadores únicos (como id_cliente) se existirem
colunas_id = [c for c in df.columns if "id" in c.lower() or "cliente" in c.lower() and c != "churn"]
if colunas_id:
    df = df.drop(columns=colunas_id)

# Tratamento simples de nulos caso existam
df = df.dropna()

# Separação de features e target
X = df.drop(columns=["churn"])
y = df["churn"]

# Identificação de colunas numéricas e categóricas
colunas_categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()
colunas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()

# One-Hot Encoding nas categóricas
X = pd.get_dummies(X, columns=colunas_categoricas, drop_first=True)

# ==========================================================
# 3. SEPARE OS DADOS EM TREINO E TESTE (70% / 30%)
# ==========================================================
print("\n" + "=" * 65)
print("3. DIVISÃO EM TREINO (70%) E TESTE (30%)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Conjunto de Treino: {X_train.shape[0]} amostras")
print(f"Conjunto de Teste:  {X_test.shape[0]} amostras")

# Padronização das variáveis numéricas
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if colunas_numericas:
    X_train_scaled[colunas_numericas] = scaler.fit_transform(X_train[colunas_numericas])
    X_test_scaled[colunas_numericas] = scaler.transform(X_test[colunas_numericas])

# ==========================================================
# 4. TREINE OS MODELOS (BASELINE, RANDOM FOREST, XGBOOST)
# ==========================================================
print("\n" + "=" * 65)
print("4. TREINAMENTO DOS MODELOS")
print("=" * 65)

modelos = {
    "Regressao Logistica (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
}

if tem_xgboost:
    modelos["XGBoost"] = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, eval_metric="logloss", random_state=42)
else:
    modelos["XGBoost (HistGradientBoosting)"] = HistGradientBoostingClassifier(random_state=42)

resultados = {}
probabilidades = {}

for nome, modelo in modelos.items():
    # Modelos lineares beneficiam de dados escalados
    if "Logistica" in nome:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

    resultados[nome] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1-score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc_score": roc_auc_score(y_test, y_prob)
    }
    probabilidades[nome] = (y_pred, y_prob)

# ==========================================================
# 5. AVALIE OS RESULTADOS
# ==========================================================
print("\n" + "=" * 65)
print("5. TABELA COMPARATIVA DE PERFORMANCE")
print("=" * 65)

df_metricas = pd.DataFrame(resultados).T
print(df_metricas.to_string(formatters={
    "accuracy": "{:.4f}".format,
    "precision": "{:.4f}".format,
    "recall": "{:.4f}".format,
    "f1-score": "{:.4f}".format,
    "roc_auc_score": "{:.4f}".format
}))

# Selecionar o melhor modelo com base em ROC-AUC
melhor_modelo_nome = df_metricas["roc_auc_score"].idxmax()
y_pred_melhor, y_prob_melhor = probabilidades[melhor_modelo_nome]

# Plot da Matriz de Confusão do melhor modelo
fig, ax = plt.subplots(figsize=(5.5, 4.5))
cm = confusion_matrix(y_test, y_pred_melhor)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ativo (0)", "Cancelou (1)"])
disp.plot(cmap="Blues", ax=ax, values_format="d")
plt.title(f"Matriz de Confusão: {melhor_modelo_nome}")
plt.tight_layout()
plt.savefig("matriz_confusao.png")
plt.close()

# Plot da Curva ROC comparativa
plt.figure(figsize=(7, 5))
for nome in modelos.keys():
    _, y_prob = probabilidades[nome]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = resultados[nome]["roc_auc_score"]
    plt.plot(fpr, tpr, label=f"{nome} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Aleatório (AUC = 0.50)")
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)")
plt.title("Curva ROC Comparativa")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("curva_roc.png")
plt.close()

# Importância das Variáveis (Feature Importance) via Random Forest
rf_model = modelos["Random Forest"]
importancias = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
importancias.sort_values().plot(kind="barh", color="#2ecc71", edgecolor="black")
plt.title("Top 10 Variáveis com Maior Impacto no Churn (Random Forest)")
plt.xlabel("Importância Relativa")
plt.tight_layout()
plt.savefig("importancia_variaveis.png")
plt.close()

# ==========================================================
# 6. INTERPRETAÇÃO E INSIGHTS DE NEGÓCIO
# ==========================================================
print("\n" + "=" * 65)
print("6. INTERPRETAÇÃO E RECOMENDAÇÕES ESTRATÉGICAS")
print("=" * 65)
print(f"Melhor modelo avaliado: {melhor_modelo_nome}")
print("\nVariáveis de maior peso identificadas:")
for rank, (var, peso) in enumerate(importancias.head(5).items(), start=1):
    print(f" {rank}. {var} (Impacto: {peso:.4f})")

print("\nInsights de Produto e Ações Recomendadas:")
print(" - Foco em Retenção Antecipada: Clientes com queda acentuada em logins e volume de ações")
print("   devem disparar alertas no sistema para contacto proativo do time de Customer Success.")
print(" - Resolução Rápida de Suporte: Reduzir tempo de resolução de chamados, pois clientes com")
print("   muitos tickets abertos tendem a concentrar a maior probabilidade de cancelamento.")
print(" - Estratégia de Pagamento: Monitorizar inadimplências precoces oferecendo flexibilização de")
print("   planos antes que o cliente atinja o cancelamento definitivo.")
