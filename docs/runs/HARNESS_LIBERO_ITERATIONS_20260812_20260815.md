# Avanços do Harness VLA em LIBERO nativo — 12 a 15/08/2026

Classificação: **beta-only** (instrumentação e calibração locais; nenhum item
abaixo alega fidelidade a mecanismo de arXiv:2607.08448v2 salvo indicação).

## Resumo executivo

| Bateria | Condição | Resultado |
|---|---|---|
| frozen_v1 (24 ep, 3 seeds) | harness original | 29,2% (7/24) |
| frozen_v1 (24 ep, 3 seeds) | baseline VLA puro | 50,0% (12/24) — spatial 12/12, object 0/12 |
| frozen_v1 (24 ep, 3 seeds) | **harness + iter 1-2** | **91,7% (22/24)** |
| probe_v2 (8 ep, goal+long) | baseline VLA puro | 25,0% (2/8) |
| probe_v2 (8 ep, goal+long) | harness + iter 1-2 | 50,0% (4/8) |
| probe_v2 (8 ep, goal+long) | **harness + iter 3** | **75,0% (6/8)** |

Pareamento sempre por (task_suite, task_id, initial_state_index, seed).

## Iterações

### Iteração 1 — contrato do `target` no prompt (commit `a66399f`)
- Diagnóstico: 16/24 episódios da frozen_v1 abortados por `holding_incompatible`;
  o planner passava o **destino** (basket/plate) no campo `target` de
  `vla_act tau=task_success`, mas o guard exige o objeto seguro.
- Correção: prompt do sistema fixa o contrato com exemplo explícito
  certo/errado.
- Efeito: 0 violações de guard em todas as baterias seguintes.

### Iteração 2 — guards recuperáveis (commit `1dfe1a4`)
- Antes: qualquer violação de guard encerrava o episódio no ato.
- Agora: a violação vira feedback por turno e o planner pode corrigir
  (limite: `max_turns`). Aumenta a agência do planner; mantém o bloqueio da
  execução insegura.

### Iteração 3 — fixtures e anti-perseveração (commit `d3343b2`)
- Diagnóstico (probe_v2): em "open the middle drawer" o planner repetiu 12
  turnos `lift_and_grasp` sobre a região do armário (`tau_setup_error`), pois o
  contrato não permitia `tau=task_success` de mão vazia.
- Correções: guard permite `tau=task_success` com mão vazia (manipulação de
  fixture: gaveta, fogão); prompt orienta fixtures, sequenciamento
  multi-objeto e proíbe repetir chamada que falhou 2× com o mesmo erro.
- Efeito pareado: gaveta 0/2 → 2/2; probe_v2 50% → 75%.

### Iteração 4 — sequenciamento multi-objeto + ablação think (commit `94a1159`)
- Diagnóstico: nas 2 falhas restantes (libero_10 "both … in the basket") o
  planner segurava o 1º objeto chamando `tau=task_success` por 300+ passos —
  o τ nunca dispara com a tarefa incompleta.
- Correções: prompt manda colocar objetos não-finais com
  `move_to release_pose` + `release` (primitivas analíticas existentes) e
  reservar `tau=task_success` ao objeto final; flag `--planner-think on|off`
  para medir custo/benefício do thinking (~70 s/turno, 73% do tempo total).
- Verificação: fila de baterias de 15/08 (abaixo).

## Fila de baterias de 15/08 (script `scripts/run_battery_queue_20260815.sh`)

1. `battery_probe_v2_harness_fix4_20260815` — verifica multi-objeto (8 ep).
2. `battery_frozen_v1_harness_fix4_20260815` — regressão da frozen_v1 (24 ep);
   meta ≥ 91,7%.
3. `battery_probe_v3_harness_20260815` + `battery_probe_v3_baseline_20260815`
   — escada nível 4: libero_90 (stacking, botão, composto, relacional,
   compartimento; 5 ep cada).
4. `battery_frozen_v1_harness_fix4_nothink_20260815` — ablação `think=off`
   (8 ep, seed 101): mede perda de sucesso vs ganho de latência.

Resultados serão anexados aqui ao final da fila.

## Lacunas conhecidas

- Multi-objeto dependia de orientação de prompt; se a iter 4 não bastar, o
  próximo candidato é um τ de subobjetivo (exige verificação prévia no paper).
- Tarefas ocultas (frozen_v1: spatial 7, object 8; probe_v3 análogas) intactas
  para avaliação final anti-overfitting via `--hidden-only`.
- `holding` pode ficar obsoleto se o VLA soltar o objeto sem `release`
  analítico; ainda sem evidência de dano, monitorar nos traces.

## Reprodução

Servidor pi0.5: porta 8010, GPU 5, eager (V100), nohup
(`evaluation_runs/serve_pirlinf_nohup.log`). Baterias: ver script da fila.
Testes: 440 passing (`pytest tests/`).
