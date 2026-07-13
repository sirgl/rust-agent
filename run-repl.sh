#!/usr/bin/env bash
# Запуск интерактивного терминального чата (REPL) агента.
#
# Использование:
#   ./run-repl.sh              # release-сборка (быстрее)
#   ./run-repl.sh --debug      # debug-сборка (быстрее компиляция)
#
# Дополнительные аргументы после флага сборки передаются в cli-agent.
# Ключ Anthropic по умолчанию берётся из token.properties (KEY=...).
set -euo pipefail

# Каталог, в котором лежит скрипт (корень проекта).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROFILE_FLAG="--release"
if [[ "${1:-}" == "--debug" ]]; then
  PROFILE_FLAG=""
  shift
fi

exec cargo run ${PROFILE_FLAG} -q -p cli-agent -- "$@"
