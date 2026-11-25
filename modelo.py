"""Exercício de aprendizado não supervisionado com dados simulados."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Dados simulados de sensores agrícolas.
    dados = pd.DataFrame({
        "umidade": [20, 22, 65, 68, 40, 42],
        "temperatura": [34, 33, 24, 23, 29, 28],
    })

    # 2. Padronização das características originais.
    X = dados[["umidade", "temperatura"]]
    X_padronizado = StandardScaler().fit_transform(X)

    # 3. Clusterização por centroides.
    modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
    dados["cluster"] = modelo.fit_predict(X_padronizado)

    # 4. Clusterização por densidade e identificação de ruído.
    modelo_db = DBSCAN(eps=0.3, min_samples=2)
    dados["dbscan"] = modelo_db.fit_predict(X_padronizado)

    # 5. Redução de dimensionalidade.
    pca = PCA(n_components=1)
    dados["PC1"] = pca.fit_transform(X_padronizado)[:, 0]

    # 6. Resultados antes de abrir a janela do gráfico.
    print("Dados processados:")
    print(dados.to_string(index=False))
    print("\nPontos de ruído (DBSCAN):")
    outliers = dados[dados["dbscan"] == -1]
    print(outliers.to_string(index=False) if not outliers.empty else "Nenhum.")
    print(f"\nVariância explicada pelo PCA: {pca.explained_variance_ratio_[0]:.2%}")

    # 7. Visualização e exportação do gráfico de K-Means.
    pasta = Path(__file__).resolve().parent / "resultados"
    pasta.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(dados["umidade"], dados["temperatura"], c=dados["cluster"], cmap="viridis")
    ax.set(title="Clusterização de sensores agrícolas (K-Means)", xlabel="Umidade (%)", ylabel="Temperatura (°C)")
    ax.grid(alpha=0.2)
    fig.savefig(pasta / "clusterizacao.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
