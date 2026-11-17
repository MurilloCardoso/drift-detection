# Drift Detect — Statlog German Credit

Pipeline local de validação de dados, classificação de risco de crédito, detecção de *data drift* e monitoramento. O projeto foi desenvolvido para o Tech Challenge MLET, Fase 4, com o conjunto público [Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog-german-credit-data).

> **Escopo:** demonstração acadêmica com dados públicos e produção simulada. O modelo e os alertas não devem ser usados para decisões reais de crédito.

## O que o projeto faz

1. **Valida os dados:** aplica um contrato Pandera à base de referência e mostra a rejeição de um lote com erros intencionais.
2. **Treina um baseline:** ajusta uma regressão logística e registra métricas em um conjunto de teste separado.
3. **Simula e detecta drift:** altera duas variáveis de um lote sintético, gera predições e compara as distribuições com Evidently, PSI e KS.
4. **Monitora a execução:** grava logs e métricas locais e gera um dashboard HTML.

## Início rápido

**Pré-requisitos:** Python 3.11, 3.12 ou 3.13 e [`uv`](https://docs.astral.sh/uv/) instalado. Execute os comandos na **raiz do repositório**, pois os caminhos de entrada e saída são resolvidos a partir do diretório atual.

```bash
uv sync --extra dev
uv run drift-detect demo
uv run drift-detect drift
uv run drift-detect monitor
```

Ao concluir, abra `artifacts/reports/statlog_drift_report.html` para a análise de drift e `artifacts/reports/statlog_monitoring_dashboard.html` para o monitoramento. Os relatórios JSON e CSV ficam nos mesmos diretórios indicados em [Arquivos gerados](#arquivos-gerados).

`demo` prepara e valida os dados, treina o modelo e demonstra a rejeição do lote inválido. `drift` cria o lote de produção simulado e gera os relatórios; se o modelo ainda não existir, ele é treinado antes. `monitor` executa validação, verifica ou treina o modelo, analisa drift e registra a execução. Assim, cada comando pode ser executado isoladamente; a sequência acima mostra o fluxo completo.

## Dados e contrato

A única fonte de referência é [`dataset/statlog/german.data`](dataset/statlog/german.data), incluída no repositório. O pipeline lê diretamente esse arquivo, sem precisar baixar ou extrair o ZIP da UCI. São **1.000 registros**, **20 atributos** e a coluna alvo `credit_risk`: `1` representa bom risco e `2`, mau risco. `record_id` é criado a partir da posição da linha no arquivo para identificar registros nesta execução; não é um identificador pessoal.

O contrato `statlog_german_credit_contract_v1` exige as 20 colunas de atributos, `credit_risk` e `record_id`; verifica tipos, limites numéricos, códigos categóricos permitidos, unicidade do ID e ausência de linhas idênticas. CSVs fornecidos ao comando `validate` devem conter essas **22 colunas** com os nomes esperados. Para o arquivo `.data` original, o leitor atribui os nomes e cria `record_id` automaticamente.

O comando `demo` cria `data/invalid/statlog_invalid_batch.csv` com erros independentes, tenta validá-lo e registra as violações em `artifacts/reports/statlog_invalid_failure_cases.csv`. A rejeição desse lote é o resultado esperado da demonstração.

Arquivos legados de uma versão anterior ainda podem aparecer em `data/` e `artifacts/`. O pipeline atual usa somente a fonte Statlog e arquivos novos com prefixo `statlog_`; também não combina `german.data-numeric` com `german.data`.

## Etapas e critérios

### 1. Classificação supervisionada

O baseline é uma **regressão logística** com imputação de valores ausentes, padronização das variáveis numéricas e codificação *one-hot* das categóricas. A classe positiva é **`credit_risk = 2` (mau risco)**. A divisão é estratificada em 80% para treino e 20% para teste, com semente `42`; o pré-processamento e o classificador são ajustados somente nos 800 registros de treino.

As métricas de teste são acurácia, acurácia balanceada, F1 e ROC AUC. No relatório incluído no repositório, elas são, respectivamente, **0,780**, **0,710**, **0,593** e **0,804** (valores arredondados). Consulte `artifacts/reports/statlog_baseline_metrics.json` para os valores completos e a execução correspondente. O modelo é salvo em `artifacts/model/statlog_credit_classifier.joblib`, arquivo ignorado pelo Git e reconstruído por `demo` ou `train`.

### 2. Simulação e detecção de drift

O lote de produção é uma cópia da referência com duas alterações controladas:

```text
age           = round(min(100, age * 1.20 + 5))
credit_amount = round(min(2000000, credit_amount * 1.80 + 1000))
```

As demais variáveis e o alvo histórico são preservados. O lote alterado passa novamente pelo contrato. O modelo gera `bad_risk_probability` e `predicted_risk`; o relatório Evidently compara as distribuições de referência e produção. O resumo adicional calcula **PSI para variáveis numéricas e categóricas** e **KS para variáveis numéricas**. Uma coluna entra na lista de alertas quando `PSI ≥ 0,1` ou `p < 0,05` no KS. O relatório Evidently usa PSI com limite `0,1` e considera drift global quando pelo menos **5% das 22 colunas analisadas** apresentam drift. `record_id` e `credit_risk` ficam fora dessa comparação; as duas saídas do modelo entram nela.

O resumo salvo no repositório sinaliza `age`, `credit_amount` e `bad_risk_probability`. As duas primeiras foram alteradas diretamente; a última reflete uma mudança na distribuição das probabilidades previstas. Esse resultado demonstra *data drift* e mudança nas predições. Como não há novos desfechos observados, ele **não mede concept drift nem degradação real de desempenho em produção**.

### 3. Observabilidade local

`monitor` registra o status, a duração e o volume das etapas de validação, prontidão do modelo e análise de drift. Também registra quantidade de alertas, proporção de predições de mau risco, espaço livre no disco e tempo de CPU do processo. O dashboard mostra a última execução, os resultados por etapa, os alertas e até 20 execuções recentes; o histórico completo permanece em JSONL. As métricas de infraestrutura descrevem a máquina local, não a disponibilidade de um serviço publicado.

## Comandos disponíveis

| Comando | Finalidade |
| --- | --- |
| `uv run drift-detect prepare` | Valida a referência e cria o lote inválido de demonstração. |
| `uv run drift-detect validate dataset/statlog/german.data` | Valida o arquivo original pelo contrato. Aceita também um CSV Statlog normalizado. |
| `uv run drift-detect train` | Treina o baseline com a referência e salva modelo e métricas. |
| `uv run drift-detect train --data caminho/para/lote.csv` | Treina com outro CSV que passe pelo mesmo contrato. |
| `uv run drift-detect demo` | Executa toda a etapa de contrato e treinamento, incluindo a rejeição esperada do lote inválido. |
| `uv run drift-detect drift` | Gera produção simulada, predições e relatórios de drift. |
| `uv run drift-detect monitor` | Executa o pipeline monitorado e atualiza logs, histórico e dashboard. |

Use `uv run drift-detect --help` para ver as opções e `uv run drift-detect --verbose <comando>` para ativar logs de depuração. Uma validação rejeitada pelo contrato retorna código de saída `2`; erros de arquivo ou de execução tratados pela CLI retornam `1`.

## Arquivos gerados

| Caminho | Conteúdo |
| --- | --- |
| `data/invalid/statlog_invalid_batch.csv` | Lote com erros criado por `prepare` ou `demo`. |
| `artifacts/reports/statlog_validation_report.json` | Resultado da validação da referência em `demo`. |
| `artifacts/reports/statlog_invalid_failure_cases.csv` | Violações do lote inválido em `demo`. |
| `artifacts/reports/statlog_baseline_metrics.json` | Métricas do teste em `train` ou `demo`. |
| `artifacts/model/statlog_credit_classifier.joblib` | Pipeline treinado; ignorado pelo Git. |
| `data/production/statlog_drifted.csv` | Lote sintético validado. |
| `artifacts/predictions/statlog_production_predictions.csv` | Probabilidades e classes previstas no lote sintético. |
| `artifacts/reports/statlog_drift_report.html` | Relatório visual Evidently. |
| `artifacts/reports/statlog_evidently_drift_report.json` | Dados do relatório Evidently. |
| `artifacts/reports/statlog_drift_summary.json` | PSI, KS, alertas e taxas de predição. |
| `artifacts/observability/statlog_pipeline.jsonl` | Logs estruturados da CLI. |
| `artifacts/observability/statlog_metrics_history.jsonl` | Histórico das execuções monitoradas. |
| `artifacts/reports/statlog_monitoring_dashboard.html` | Dashboard HTML local. |

## Qualidade e reprodução

```bash
uv run pytest
uv run ruff check .
```

O [workflow de CI](.github/workflows/ci.yml) executa testes, lint e os comandos `demo`, `drift` e `monitor` em Python 3.12, depois publica relatórios, logs e predições como artefatos da execução. Os resultados locais podem variar se as dependências ou os dados de entrada forem alterados.

## Governança e limites de uso

O conjunto de dados é público, mas atributos de crédito vinculáveis a indivíduos exigem tratamento adequado de dados pessoais. Este protótipo não implementa controles de acesso, criptografia, política de retenção ou revisão humana para uso em produção. Para dados reais, seria necessário definir finalidade e base legal, reduzir os atributos ao necessário, pseudonimizar identificadores, restringir acessos, proteger dados em trânsito e em repouso, estabelecer retenção e descarte e oferecer canais para exercício de direitos e contestação de decisões automatizadas. A aplicação da [LGPD](https://legis.senado.gov.br/norma/27457334/publicacao/27457731) e das [orientações da ANPD](https://www.gov.br/anpd/pt-br/assuntos/titular-de-dados/direito-dos-titulares) deve ser avaliada para o caso concreto.

A intervenção nas duas variáveis causa o drift medido **neste lote sintético**. Ela não demonstra que inflação real altera risco de crédito, nem que a qualidade de decisões cairia em um sistema real. Essas conclusões exigiriam desfechos futuros observados, desenho temporal e controle de fatores de confusão.

## Apresentação STAR

**Vídeo:** pendente de gravação e publicação pelo grupo. Roteiro sugerido, de até cinco minutos: **situação** — risco de degradação silenciosa; **tarefa** — validar, classificar, detectar drift e discutir governança; **ação** — demonstrar `demo`, `drift` e `monitor`; **resultado** — apresentar métricas de teste, rejeição do lote inválido, alertas e limites da simulação.
