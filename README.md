# Drift Detect — Statlog German Credit

Validação de dados, classificação supervisionada, detecção de *data drift* e monitoramento local para o Tech Challenge MLET, Fase 4.

## Fonte e separação dos dados

A referência é exclusivamente `dataset/statlog/german.data`, extraído do [repositório UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog-german-credit-data) e incluído no projeto. O pipeline lê esse arquivo diretamente, sem depender do ZIP: 1.000 registros, 20 atributos e o alvo `credit_risk` (`1` = bom risco, `2` = mau risco). O `record_id` é gerado pela posição da linha apenas para identificar registros nesta execução; não identifica uma pessoa.

O CSV usado na versão anterior foi removido. Seus artefatos legados ainda podem existir no checkout, mas **não são unidos, usados no treinamento ou validados pelo contrato Statlog**. Relatórios, logs, modelo e lotes novos têm prefixo `statlog_`. O arquivo `german.data-numeric` também não é combinado ao original.

## Execução

Requer Python 3.11–3.13 e `uv`:

```bash
uv sync --extra dev
uv run drift-detect demo
uv run drift-detect drift
uv run drift-detect monitor
uv run pytest
uv run ruff check .
```

Os scripts `scripts/run_stage1`, `run_stage2` e `run_stage3` executam as mesmas etapas. O CI executa os três comandos, testes e lint, e publica as evidências geradas.

## Etapa 1 — contrato e classificador

O contrato Pandera `statlog_german_credit_contract_v1` exige as 20 features oficiais, `credit_risk` em `{1, 2}`, ID interno único, tipos e faixas numéricas, códigos categóricos conhecidos e ausência de linhas idênticas. O lote `data/invalid/statlog_invalid_batch.csv` injeta erros independentes e é bloqueado com casos em `artifacts/reports/statlog_invalid_failure_cases.csv`.

O baseline é uma regressão logística com pré-processamento (imputação, padronização e one-hot encoding). A classe positiva é `2`, mau risco. A avaliação usa divisão estratificada 80/20 com semente fixa; o pipeline é ajustado apenas nos 800 registros de treino. Acurácia, acurácia balanceada, F1 e ROC AUC no teste ficam em `artifacts/reports/statlog_baseline_metrics.json`. O modelo persistido fica em `artifacts/model/statlog_credit_classifier.joblib` (ignorado pelo Git e reconstruído pelo comando `demo`). Esses resultados são de um exercício acadêmico, não qualificam o modelo para decisões reais.

## Etapa 2 — drift

O lote de produção simulado preserva as demais features e o alvo, mas transforma `age = round(min(100, age * 1.20 + 5))` e `credit_amount = round(min(2000000, credit_amount * 1.80 + 1000))`. Ele passa novamente pelo contrato. O classificador produz `bad_risk_probability` e `predicted_risk`; o relatório Evidently compara referência e produção, e o resumo calcula PSI para todas as variáveis e KS para as numéricas. O alerta global do Evidently usa participação mínima de 5% das colunas, suficiente para as duas features alteradas entre as 22 analisadas. `record_id` e `credit_risk` não entram no teste de drift. Veja `artifacts/reports/statlog_drift_report.html` e `artifacts/reports/statlog_drift_summary.json`.

Os rótulos preservados são apenas os desfechos históricos da referência em um cenário sintético. Não representam resultados futuros observados; portanto, a simulação demonstra *data drift* e mudança das predições, não *concept drift* nem degradação real de desempenho em produção.

## Etapa 3 — observabilidade

`monitor` registra status e duração de validação, prontidão do modelo e análise de drift; volume, alertas e taxa de predição de mau risco; espaço livre no disco e tempo de CPU do processo. Logs JSONL e histórico ficam em `artifacts/observability/statlog_*.jsonl`. O dashboard `artifacts/reports/statlog_monitoring_dashboard.html` apresenta a última execução, testes e até 20 execuções recentes. As métricas de infraestrutura são locais e não medem disponibilidade de um serviço em produção.

## Etapa 4 — governança e causalidade

Este repositório é uma demonstração acadêmica com dados públicos. As variáveis de crédito devem ser tratadas como dados pessoais quando puderem ser vinculadas a indivíduos. O projeto não deve tomar decisões reais de concessão. Em produção, o controlador deve documentar finalidade, origem, necessidade dos atributos e compartilhamentos. A hipótese de proteção do crédito do art. 7º, X, da [LGPD](https://legis.senado.gov.br/norma/27457334/publicacao/27457731) exige avaliação jurídica para o caso concreto. Titulares precisam de canais para acesso, correção, contestação e revisão de decisões unicamente automatizadas, conforme a [ANPD](https://www.gov.br/anpd/pt-br/assuntos/titular-de-dados/direito-dos-titulares).

Plano para dados reais: pseudonimizar IDs e guardar a correspondência separadamente; limitar acessos por função; criptografar dados em trânsito e repouso; manter logs agregados; definir uma tabela de retenção por finalidade e obrigação legal, com eliminação ou anonimização após o prazo aprovado; avaliar necessidade e impacto de atributos associados a grupos protegidos, com revisão humana. Este protótipo não implementa essas salvaguardas de produção nem estabelece prazo legal universal.

A intervenção controlada nas duas features causa o drift medido no lote sintético. Isso não prova efeito da inflação real, mudança causal no risco de crédito ou piora da qualidade decisória. Essa conclusão exigiria desfechos futuros observados, desenho temporal e controle de fatores de confusão.

## Vídeo STAR

Link: **pendente de gravação e publicação pelo grupo**. Roteiro de até cinco minutos: situação (risco de degradação silenciosa), tarefa (contrato, classificação, drift e governança), ação (`demo`, `drift`, `monitor`), resultado (métricas de teste, rejeição do lote inválido, alertas e limites da simulação).
