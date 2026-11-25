# ML — Clusterização de Sensores Agrícolas

Versão inicial: **V1.0**.

Exercício de aprendizado não supervisionado desenvolvido no contexto da Fase 4, Capítulo 09 do curso de Inteligência Artificial da FIAP. O objetivo é explorar padrões em leituras simuladas de umidade do solo e temperatura utilizando Python e scikit-learn.

## Metodologia

1. Criação de seis observações simuladas, sem rótulos prévios.
2. Padronização das duas características com `StandardScaler`.
3. K-Means com três clusters, `random_state=42` e `n_init=10`.
4. DBSCAN com `eps=0.3` e `min_samples=2`, aplicado aos mesmos dados padronizados.
5. PCA com um componente principal.
6. Exibição dos resultados e exportação do gráfico de K-Means.

## Estrutura

| Arquivo | Finalidade |
| --- | --- |
| `modelo.py` | Base simulada, algoritmos e visualização |
| `requirements.txt` | Dependências Python |
| `.gitignore` | Arquivos locais e saídas geradas |
| `resultados/clusterizacao.png` | Gráfico gerado ao executar o programa (não versionado) |

## Como executar

Requer Python 3.13 ou ambiente compatível com as dependências. A execução foi verificada com Python 3.13.5, pandas 2.2.3, Matplotlib 3.10.8 e scikit-learn 1.8.0, em ambiente Linux.

```bash
git clone https://github.com/Caiobqz/ML-Clusterizacao-Sensores-Agricolas.git
cd ML-Clusterizacao-Sensores-Agricolas
python -m venv .venv
```

Ative o ambiente no Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux/macOS: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python modelo.py
```

Os resultados são impressos antes da abertura do gráfico, que também é salvo na pasta `resultados/`.

## Resultados da execução de referência

| Método | Resultado |
| --- | --- |
| K-Means | 3 grupos |
| DBSCAN | 3 grupos e nenhum ponto de ruído |
| PCA | 2 variáveis reduzidas para 1 componente; 99,79% de variância explicada |

Os dois agrupamentos identificam os mesmos pares de observações neste exemplo: solo mais seco/quente, mais úmido/frio e condições intermediárias. Os números dos clusters são identificadores arbitrários e podem variar conforme o ambiente.

O percentual de variância explicada não representa acurácia. O rótulo -1 do DBSCAN indica ruído segundo os parâmetros escolhidos, não necessariamente erro de sensor.

## Limitações e próximos passos

A base contém apenas seis registros fictícios; os resultados não validam um sistema agrícola real nem sustentam decisões de irrigação. O número de clusters e os parâmetros foram escolhidos para fins didáticos. Futuras versões poderão ampliar a base, testar anomalias e comparar configurações com o método do cotovelo e a silhueta.

## Referências

- FIAP, Fase 4, Capítulo 09 — Machine learning sem supervisão: uma jornada pela descoberta de dados.
- [Documentação do scikit-learn](https://scikit-learn.org/stable/)
- [Documentação do pandas](https://pandas.pydata.org/docs/)
- [Documentação do Matplotlib](https://matplotlib.org/stable/)
