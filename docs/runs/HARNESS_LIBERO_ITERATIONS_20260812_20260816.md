# Avanços do Harness VLA em LIBERO nativo — 12 a 16/08/2026

Classificação: **beta-only** (instrumentação e calibração locais; nenhum item
abaixo alega fidelidade a mecanismo de arXiv:2607.08448v2 salvo indicação).

## Resumo executivo

| Bateria | Condição | Resultado |
|---|---|---|
| frozen_v1 (24 ep, 3 seeds) | harness original | 29,2% (7/24) |
| frozen_v1 (24 ep, 3 seeds) | baseline VLA puro | 50,0% (12/24) — spatial 12/12, object 0/12 |
| frozen_v1 (24 ep, 3 seeds) | harness + iter 1-2 | 91,7% (22/24) |
| frozen_v1 (24 ep, 3 seeds) | **harness + iter 1-5 (regressão)** | **95,8% (23/24)** |
| probe_v2 (8 ep, goal+long) | baseline VLA puro | 25,0% (2/8) |
| probe_v2 (8 ep, goal+long) | harness + iter 1-2 | 50,0% (4/8) |
| probe_v2 (8 ep, goal+long) | **harness + iter 3-6** | **75,0% (6/8)** |
| probe_v3 (5 ep, libero_90) | baseline VLA puro | 20,0% (1/5) |
| probe_v3 (5 ep, libero_90) | **harness + iter 1-6** | **60,0% (3/5)** |
| frozen_v1 seed 101 (8 ep) | harness think=on | 8/8; planner 91 s/turno (média) |
| frozen_v1 seed 101 (8 ep) | harness think=off | 7/8; planner 36 s/turno (média) |

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
- Verificação: traces confirmam a mecânica grasp → release_pose → release,
  mas o planner re-agarrava o objeto já colocado (sem memória) → iter 6.

### Iteração 5 — retries de LLM e parse recuperável (commit `3d2fc22`)
- Diagnóstico: falhas de infraestrutura (resposta vazia do Ollama e timeout de
  socket a 600 s — gemma4:12b roda 100% em CPU) abortavam episódios.
- Correções: `_chat` com 3 tentativas e backoff; `planner_parse_error` vira
  feedback por turno em vez de abortar (mesmo padrão da iter 2).
- Efeito: 0 falhas de infraestrutura em todas as baterias seguintes;
  regressão frozen_v1 subiu para 95,8% (23/24; única falha estocástica de
  horizonte em object task00 state01 seed 103).

### Iteração 6 — memória simbólica `turn_history` (commit `7c6b0b1`)
- Diagnóstico: sem memória do que já fez, o planner re-agarrava o objeto já
  colocado na cesta (perseveração entre turnos).
- Correção: o estado semântico passa a incluir `turn_history` (primitiva,
  target, modo/τ, resultado e `holding` após cada turno — simbólico, sem
  pixels); prompt instrui a tratar objeto com `release_completed_task_incomplete`
  como concluído.
- Efeito verificado nos traces da fila v3: sequência correta
  sopa → cesta → release → **passa ao molho de tomate sem re-agarrar**.
  A falha restante é física (VLA congelado não consegue o grasp do 2º objeto),
  não lógica.

## Fila de baterias v3 de 16/08 (script `scripts/run_battery_queue_v3_20260816.sh`)

Concluída em 2026-08-16T06:48Z sem retries; completude verificada contra os
manifests (0 falhas de infraestrutura).

| Bateria | Resultado | Observações |
|---|---|---|
| `battery_probe_v2_harness_fix6_20260816` | 6/8 (75%) | 2 falhas: libero_10 task00 (grasp físico do 2º objeto); lógica multi-objeto correta |
| `battery_probe_v3_harness_fix6_20260816` | 3/5 (60%) | sucessos: botão do fogão, relacional, livro no compartimento; falhas: stacking e fogão+panela (horizonte) |
| `battery_probe_v3_baseline_20260816` | 1/5 (20%) | só o botão do fogão |
| `battery_frozen_v1_harness_fix6_nothink_20260816` | 7/8 (87,5%) | think=off; planner 36 s/turno vs 91 s/turno com think (2,5× mais rápido, −1 sucesso) |

## Lacunas conhecidas

- Multi-objeto: lógica resolvida (iter 4+6); gargalo agora é o grasp físico do
  VLA congelado no 2º objeto — candidato: re-tentativa com prompt de grasp
  alternativo ou `move_to` de aproximação antes do `vla_act`.
- libero_90 stacking (task 17) e composto fogão+panela (task 21) esgotam
  horizonte 400; investigar traces antes de propor mudança.
- Tarefas ocultas (frozen_v1: spatial 7, object 8; probe_v3 análogas) intactas
  para avaliação final anti-overfitting via `--hidden-only`.
- Ablações de memória e mais seeds nas sondas (hoje: 1 seed) pendentes.
- `holding` pode ficar obsoleto se o VLA soltar o objeto sem `release`
  analítico; ainda sem evidência de dano, monitorar nos traces.

## Reprodução

Servidor pi0.5: porta 8010, GPU 5, eager (V100), nohup
(`evaluation_runs/serve_pirlinf_nohup.log`). Baterias: ver scripts das filas
(`scripts/run_battery_queue_v3_20260816.sh`).
Testes: 441 passing (`pytest tests/`).
