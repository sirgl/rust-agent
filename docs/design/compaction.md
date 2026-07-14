# Compaction в Matterhorn: детальный разбор механизма

Документ детально описывает, как устроена компакция (сжатие контекста) в `matterhorn_core`: какие уровни существуют, какие именно промпты используются, и какие крайние случаи (edge cases) обрабатываются. В конце — предложение по переносу в `rust-agent`.

> Важно: речь идёт о компакции **истории диалога** (conversation/observations), а не о системе долговременной памяти (`MemoryManager`). Memory-компактор упоминается только как отдельный частный случай (см. §7).

> Статус реализации: базовый вариант перенесён в код — модуль `crates/agent-core/src/compaction.rs` (трейт `Compactor`, seam `TextSummarizer`, `HeuristicCompactor`, `LlmCompactor`, `ChainedCompactor`). Он реализует §8 (эвристика + LLM + каскад) с сохранением парности `tool_call`/`tool_result` и fallback на оригинал. Кэш-осведомлённость (§2.2) и структурные блоки уровня B (§3.1) пока не перенесены.

## 1. Общая картина: три уровня компакции

В Matterhorn компакция выполняется на трёх независимых уровнях, срабатывающих в разных ситуациях:

| Уровень | Класс | Когда срабатывает | Использует LLM? |
|---------|-------|-------------------|-----------------|
| **A. Внутришаговый (in-turn)** | `UnlimitedStepsHistoryProcessor` | Во время работы над одной задачей, когда история распухла | Нет (чистая эвристика) |
| **B. Межзадачный (between-task)** | `CompressHistoryProcessor` + `MainAgentHistoryCompressor` | После завершения задачи, перед следующей | Да (суммаризация команд) |
| **C. On-demand / hard** | `HistoryCompressionManager.runOnDemandCompression` | По явной команде `/compact` пользователя | Нет (жёсткая обрезка) |

Оркестратор — `HistoryCompressionManager.dispatchHistoryCompression()`, который вызывается из `MainAgent`:

```
compressBeforeNextTask == true            -> runOnDemandCompression() (жёсткое, синхронно)
!isProcessed && последняя задача finished -> launchCompressHistory() (фоново, coroutine)
```

Фоновая компакция (B) запускается в `coroutineScope.launch` и её результат применяется атомарно через `stateHolder.update { ... isProcessed = true }`. Есть `awaitCompletion()` для graceful-ожидания.

## 2. Уровень A — `UnlimitedStepsHistoryProcessor` (эвристика)

Это самый частый и дешёвый уровень. Работает поверх `List<AgentObservation>` без вызовов LLM.

### 2.1. Условие срабатывания (крайние случаи!)
Компакция запускается НЕ всегда. В начале стоит защитный «ранний выход»:

```kotlin
if ((observationsSize <= KEEP_TAIL || (maxStep - step) < 20)
    && observationsLength <= TOO_BIG_OBSERVATION * hugeLimitCoef
    && !isCacheDead) {
  return /* без изменений */
}
```

- `KEEP_TAIL = 10` — если шагов мало, не трогаем.
- `(maxStep - step) < 20` — если до лимита шагов осталось мало, не тратим ресурсы на сжатие (задача скоро закончится).
- `TOO_BIG_OBSERVATION = 100_000 * 3` символов — жёсткий потолок: если превышен, сжимаем в любом случае.
- Константы завязаны на провайдера: `hugeLimitCoef = 2.0` для Anthropic/Google, иначе `1.0`.

### 2.2. Кэш-осведомлённость (ключевой edge case)
Логика тесно связана с кэшированием промптов у провайдера:
- `getSecondsSinceLastCachedRequest()` даёт время с последнего кэшируемого запроса.
- `isCacheDead = timeSinceLastCached > cacheTimeout * 2` — кэш уже точно протух → сжимать выгодно (мы всё равно не сэкономим на кэше).
- `maybeCacheDead = timeSinceLastCached > cacheTimeout` — кэш под угрозой.
- `cachesAdditionalCosts = 1.25` для Anthropic (запись в кэш дороже).

Смысл: **сжимать историю имеет смысл, только если выигрыш в токенах больше, чем потеря кэша**. Отсюда сложное условие `shouldCompact`:

```kotlin
val shouldCompact = (delta > MIN_DELTA_CHARS) && (
    isCacheDead ||
    maybeCacheDead && delta > MIN_DELTA_CHARS * 2 ||
    delta > uncachedTailLength * cachesAdditionalCosts * hugeLimitCoef ||
    /* ... огромная история ... */
    observationsLength > TOO_BIG_OBSERVATION * hugeLimitCoef * 2
)
```
где `delta` — сколько символов мы сэкономим, `uncachedTailLength` — длина «хвоста», который придётся перекэшировать.

### 2.3. Что именно вырезается (Prefix/Tail)
`splitObservationForHistoryProcessing` делит историю на `prefix` (старое) и `tail` (последние N). Из **префикса** удаляются «шумные» действия, а хвост сохраняется целиком. Правила (`dropActionsFromPrefixKeepingTail`), с индивидуальным `keepTail` на группу:

- `search_project, search_paths_by_glob, search_contents_by_grep, list_directory` → keepTail **10**
- `run_test, build, bash, powershell` → keepTail **10**
- `open, get_file_structure, open_entire_file, scroll_up, scroll_down` → keepTail **15**
- `update_status` → keepTail **100**
- `execute_step, review_step` → keepTail **80**
- `deduplicateCreateActionsInTail(5)` — дедупликация `create` по имени файла (оставляем только последнее создание того же файла).
- `dropObserverPlanRequests()` — выкидывание запросов плана наблюдателя.

### 2.4. Тонкие крайние случаи уровня A
- **Целостность tool-use**: при вырезании записей `rebuildWithRetainedRecords` пересобирает `MatterhornAssistantChatMessageWithToolUses`, синхронно фильтруя `toolUses` по оставшимся `toolCallId`. Если после фильтрации остаётся 0 записей — вся observation удаляется (`return null`); если хоть один retained-record не является tool-request — observation оставляется как есть (нельзя частично ломать связку call/result).
- **Сохранение кэша**: через `findCommonStartPart` вычисляется общий неизменённый префикс, чтобы не «сдвинуть» границу кэша без нужды.
- **Language tag**: если после сжатия из истории пропал `<LANGUAGE_TAG>`, выставляется `shouldResetLanguageInstruction = true`, чтобы вернуть языковую инструкцию.

## 3. Уровень B — `CompressHistoryProcessor` (LLM-суммаризация)

Полное сжатие истории завершённой задачи. Формирует новый компактный список синтетических observations. Это то, что вы видите в начале сессии как блоки `History processor: ...`.

### 3.1. Структура результата
`processObservations` собирает observations в фиксированном порядке:
1. **`PROCESSED_START`** — синтетическое вводное сообщение + языковая инструкция:
   > "The current session included prior operations, but the history has been compressed to retain only the essential information and last messages."
2. **Previous tasks** (`PreviousTasksProcessor`) — сжатые прошлые задачи (см. §4).
3. **Subagents** (`SubagentsProcessor`) — сводка по сабагентам (см. §5).
4. **Shown code** (`printAllCode`) — актуальный код просмотренных файлов.
5. **Changes** (`collectChanges`) — git-diff внесённых изменений.
6. **Summarized commands** (`summarizeOtherCommands`) — LLM-сводка команд (см. §3.3).
7. **Tail** — последние `NUM_MESSAGES_TO_SHOW = 2` наблюдения «как есть».

### 3.2. Выбор хвоста (edge cases)
```kotlin
val lastObservations = when {
  input.observations.lastOrNull()?.hasTerminalAction() == true -> emptyList()
  !isSameMode -> emptyList()
  else -> input.observations.takeLast(NUM_MESSAGES_TO_SHOW)
    .filterNot { it.hasTerminalAction() || it.isProcessedStart() }
    .map { it.dropObserverPlanRequests() }
}
```
- Если последнее действие терминальное (`submit`/`answer`/`suggest_plan`) — хвост пуст.
- Если сменился режим агента (`isCompatibleWithPreviousTask`: другой `issueType` или `modelAndApiVersion`) — хвост тоже пуст.
- `CREATE`/`EDIT` дропаются из НЕ-последних хвостовых observations (их содержимое уже отрисовано в `printAllCode`), но сохраняются на самой последней observation, чтобы модель видела последнее изменение целиком.

### 3.3. Промпт суммаризации команд (`summarizeOtherCommands`)
Суммаризируются только «шумные» группы: `OTHER, MOBILE, RUN, BASH, MCP`. Команды с уже готовой фоновой сводкой (`record.summary`) LLM не гоняются повторно. Точный user-промпт:

```
Below is a list of commands and the system’s responses.
For each command provide the exact command and give a concise summary of the system’s response, based on the command type:
- Run Tests: Summarize the results. If any tests failed, list which ones and explain why they failed.
- Run Other Scripts: Summarize the results. If a script failed, mention the main reason.
- Software Installation: Specify which software was installed and its version. If the installation failed, state the main reason.
- Other: Give a brief summary of the response with mention of important details.

Do not include interpretations; simply summarize what you observe.
Place all information about one command on a single line. Avoid `command` and `response` in any form in your answer, just provide requested information.
Always enclose the command in backticks (`) exactly as shown below. Use format:
`some command`: Summary of the response.

<command>...</command>
<RESPONSE>...</RESPONSE>
```
System-сообщение: `"Your task is to summarize the commands and their results."`

### 3.4. Компакция самих команд (`compactIfNeeded`, edge case)
Даже сводки команд ограничены: `MAX_COMPACTED_COMMANDS = 30`. При превышении сохраняются `RECENT_COMMANDS_TO_KEEP = 10` самых свежих + топ по частоте вызова (`count`, с маркером `(×N)`); остальные отбрасываются. Счётчики `count` мёржатся между историческими, фоновыми и свежесуммаризированными командами.

### 3.5. Суммаризатор одиночного вывода (`ToolCallSummarizer`)
Отдельный компонент, который сжимает вывод одного инструмента (используется как фоновая сводка `record.summary`). Крайние случаи:
- Короткий вывод (`≤ SHORT_OUTPUT_MAX_CHARS = 100` символов и `≤ SHORT_OUTPUT_MAX_LINES = 3` строк) возвращается как есть — без LLM.
- System-промпт прямо объясняет модели, что её сводка **заменит** сырой вывод, а сама команда остаётся в истории (её не надо повторять).

### 3.6. Печать кода и diff (edge cases)
- `printAllCode`: показывает актуальные версии просмотренных файлов. Лимиты: `maxFiles = 10`, `maxLines = 3000`, `maxChars = 60_000`; при превышении вместо кода печатается только список имён файлов. Диапазоны строк объединяются, если между ними `< MERGE_CODE_THRESHOLD = 20` строк; если покрытие файла > 80% — печатается файл целиком. Исключаются созданные файлы и явно выбранные пользователем пути.
- `collectChanges`: собирает git-diff. Длинные строки (`> 3000`) обрезаются до 200 символов с маркером `... [line truncated]`. При превышении `maxLines = 500` diff режется по блокам (`trimChangesBlocks`, `maxBlocks = 5`, от новых к старым) с маркером `... [changes truncated]`.

## 4. `PreviousTasksProcessor` — сжатие прошлых задач

Каждая прошлая задача превращается в группу тегов `<previous_issue>` / `<previous_issue_solution>` / `<previous_issue_update_by_user>` / `<assistant_question>` / `<user_answer>`. Крайние случаи усечения:
- `truncatePrevTasks`: держит максимум `MAX_FORCE_COMPRESS_DESCRIPTIONS = 10` последних задач; из них полными (с решением) остаются только последние `MAX_FULL_GROUPS = 5`, у более старых остаётся лишь `<previous_issue>` + плейсхолдер `[Solution and other details were omitted...]`.
- `truncatePrevTasksKeepFirstAndLast`: сохраняет первую (самую информативную) задачу и последние `KEEP_LAST_GROUPS = 3`, середину заменяет плейсхолдером.
- Дедупликация: уже обработанный блок не дублируется (`if (!contains(block))`).
- Из summary решения вырезаются теги плана и команды (`stripPlanAndCommand`).
- `RouterHistoryProcessor` — облегчённая синхронная версия (без LLM/IO): даёт роутеру только инфо о прошлых задачах, перезаворачивая их в `<previous_task number=.. total=..>`.

## 5. `SubagentsProcessor` — сжатие работы сабагентов

Формирует секцию `<subagents>` с блоками `<subagent_run>` (task + result). Крайние случаи:
- Единица учёта — **spawn-occurrence** (по `toolCallId` вызова `spawn_subagent`), а не handle: continuation переиспользует handle, но считается отдельным запуском.
- `MAX_SUBAGENT_RUNS = 5` — хранятся только 5 последних запусков, остальные → плейсхолдер `[Earlier subagent runs were omitted...]`.
- Статус `PENDING`, пока запуск не завершён; статус привязывается к последнему pending-запуску с тем же handle.
- Есть fallback-парсер для legacy-формата статусов (`- agent-1 (name) [STATUS]`).

## 6. Уровень C — жёсткая on-demand компакция

`runOnDemandCompression` (по `/compact`) не вызывает LLM: он берёт только блок прошлых задач и прогоняет через `truncatePrevTasks`, отбрасывая всё остальное. После этого диспатчится хук `SessionStart(source = COMPACT)` и событие `ContextCompressedEvent`.

## 7. Частный случай: memory-компактор

Отдельно от истории существует `MemoryCompactor` (сжатие файла долговременной памяти). Он ближе к «классической» LLM-компакции с фоллбеком:
- Порог `COMPACTOR_THRESHOLD = 10_000`, целевая длина `COMPACTOR_TARGET_LENGTH = 5_000`, жёсткий предел `MAX_LEN = 50_000`.
- Fallback `truncateByBlocks`: если LLM вернула пусто / длиннее оригинала / упала — текст режется по блокам `\n\n` с конца до `MAX_LEN`; блоки, что сами больше `MAX_LEN`, выбрасываются.

## 8. Предложение для Rust Agent

В Rust мы переносим оба подхода — и эвристический, и LLM-based — за единым интерфейсом.

### 8.1. Единый интерфейс
```rust
#[async_trait]
pub trait Compactor {
    /// Принимает историю, возвращает её сжатую версию.
    async fn compact(&self, input: Conversation) -> Conversation;
}
```

### 8.2. Эвристический компактор (без LLM)
Благодаря типизированной истории (`HistoryEntry`) фильтрация делается через `match`, без парсинга строк:

```rust
pub fn compact_history_heuristically(history: &Conversation, keep_tail: usize) -> Conversation {
    let split = history.entries.len().saturating_sub(keep_tail);
    let (prefix, tail) = history.entries.split_at(split);

    // В префиксе оставляем только правки и ассистентские сообщения,
    // выкидывая «шумные» поиски/чтения (их результат уже использован).
    let mut entries: Vec<_> = prefix.iter().filter(|e| match e {
        HistoryEntry::ToolCall(tc) => matches!(tc.tool, KnownTool::Edit(_)),
        HistoryEntry::Assistant(_) => true,
        _ => false,
    }).cloned().collect();

    entries.extend_from_slice(tail); // хвост — как есть
    Conversation { entries }
}
```
> Важно (edge case из Matterhorn): при выкидывании `ToolCall` нужно выкидывать и парный `ToolResult` (по `ToolCallId`), иначе история станет невалидной для провайдера.

### 8.3. LLM-based компактор (умное сжатие)
Дублирует уровень B: собирает «шумные» вызовы инструментов, отдаёт их модели-суммаризатору и заменяет сырые выводы краткими сводками. Промпт переносится дословно из `ToolCallSummarizer`/`summarizeOtherCommands`.

```rust
pub struct LlmCompactor {
    client: Arc<dyn LlmClient>,
    keep_tail: usize,
    threshold_chars: usize, // не сжимаем, пока история меньше порога
}

#[async_trait]
impl Compactor for LlmCompactor {
    async fn compact(&self, input: Conversation) -> Conversation {
        if estimate_len(&input) <= self.threshold_chars {
            return input; // ранний выход, как в UnlimitedStepsHistoryProcessor
        }
        let split = input.entries.len().saturating_sub(self.keep_tail);
        let (prefix, tail) = input.entries.split_at(split);

        let mut out = Vec::new();
        for entry in prefix {
            match entry {
                // Короткий вывод оставляем как есть (SHORT_OUTPUT_MAX_*), длинный — суммаризируем.
                HistoryEntry::ToolResult(r) if should_summarize(r) => {
                    match self.summarize(r).await {
                        Ok(summary) => out.push(replace_with_summary(r, summary)),
                        Err(_) => out.push(entry.clone()), // fallback: оставляем оригинал
                    }
                }
                _ => out.push(entry.clone()),
            }
        }
        out.extend_from_slice(tail);
        Conversation { entries: out }
    }
}
```

Ключевые правила, которые надо сохранить при переносе:
- **System-промпт** прямо сообщает модели, что её сводка *заменяет* сырой вывод, а команда остаётся в истории (не повторять её).
- **Короткие выводы** (≤100 символов и ≤3 строк) — без вызова LLM.
- **Fallback обязателен**: при ошибке/пустом ответе LLM оставляем оригинал (или режем по блокам, как `truncateByBlocks`), чтобы не потерять контекст.
- **Дедуп парных tool_call/tool_result** по `ToolCallId`.

### 8.4. Каскад (Chained)
`ChainedCompactor` = сначала дешёвая эвристика (§8.2), затем, если всё ещё велико, LLM-суммаризация (§8.3), и жёсткая обрезка как последний рубеж. Это повторяет трёхуровневую модель Matterhorn.

## 9. Почему это "по красоте"

- **Простота**: Compaction — это чистый этап `Conversation -> Conversation` перед отправкой в LLM, без сложных систем состояния.
- **Надежность**: Каскад «эвристика → LLM → жёсткая обрезка» с fallback на каждом шаге исключает и переполнение контекста, и потерю данных при сбое модели.
- **Типобезопасность**: фильтрация истории по вариантам `HistoryEntry`/`KnownTool`, а не по regex; парность `tool_call`/`tool_result` контролируется по `ToolCallId`.
- **Эффективность**: эвристика бесплатна по токенам; LLM-компакция вызывается только когда выигрыш реально оправдан (порог + кэш-осведомлённость).
