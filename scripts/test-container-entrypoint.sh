#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
entrypoint=$repo_root/docker-entrypoint.sh
test_root=$(mktemp -d)
cleanup() {
    status=$?
    rm -rf "$test_root"
    exit "$status"
}
trap cleanup EXIT

mkdir -p "$test_root/bin"
printf '%s\n' \
    '#!/bin/sh' \
    'if [ "${EXPECT_DATABASE_URL_UNSET:-}" = 1 ] && [ "${DATABASE_URL+x}" = x ]; then exit 1; fi' \
    'exit 0' >"$test_root/bin/agentic-server"
chmod +x "$test_root/bin/agentic-server"
test_path=$test_root/bin:/usr/bin:/bin

unset_dir=$test_root/unset
mkdir "$unset_dir"
(
    cd "$unset_dir"
    env -u DATABASE_URL AGENTIC_API_HOME="$unset_dir" EXPECT_DATABASE_URL_UNSET=1 PATH="$test_path" "$entrypoint"
)
test -f "$unset_dir/agentic_api.db"
test "$(stat -c '%a' "$unset_dir/agentic_api.db" 2>/dev/null || stat -f '%Lp' "$unset_dir/agentic_api.db")" = "664"

sqlite_dir=$test_root/sqlite
mkdir "$sqlite_dir"
DATABASE_URL="sqlite://$sqlite_dir/state.db?mode=rwc&cache=shared#fragment" PATH="$test_path" "$entrypoint"
test -f "$sqlite_dir/state.db"
test -w "$sqlite_dir/state.db"

printf '%s' preserved >"$sqlite_dir/existing.db"
DATABASE_URL="sqlite://$sqlite_dir/existing.db" PATH="$test_path" "$entrypoint"
test "$(cat "$sqlite_dir/existing.db")" = preserved

for mode in ro rw memory; do
    case_dir=$test_root/untouched-$mode
    mkdir "$case_dir"
    (
        cd "$case_dir"
        DATABASE_URL="sqlite://$case_dir/state.db?mode=$mode" PATH="$test_path" "$entrypoint"
    )
    test -z "$(find "$case_dir" -mindepth 1 -print -quit)"
done

for database_url in 'sqlite::memory:' 'postgresql://agentic-api@postgres.example.com/agentic_api'; do
    case_dir=$test_root/untouched-$(printf '%s' "$database_url" | tr -cd '[:alnum:]')
    mkdir "$case_dir"
    (
        cd "$case_dir"
        DATABASE_URL=$database_url PATH="$test_path" "$entrypoint"
    )
    test -z "$(find "$case_dir" -mindepth 1 -print -quit)"
done

bare_dir=$test_root/bare
mkdir "$bare_dir"
chmod 700 "$bare_dir"
(
    cd "$bare_dir"
    DATABASE_URL=sqlite:// PATH="$test_path" "$entrypoint"
)
test -z "$(find "$bare_dir" -mindepth 1 -print -quit)"
test "$(ls -ld "$bare_dir" | cut -c5-7)" = "---"

encoded_dir=$test_root/encoded
mkdir "$encoded_dir"
if DATABASE_URL="sqlite://$encoded_dir/state%23prod.db" PATH="$test_path" "$entrypoint" 2>"$encoded_dir/error.log"; then
    echo "percent-encoded SQLite path unexpectedly succeeded" >&2
    exit 1
fi
grep -q "percent-encoded SQLite paths are not supported" "$encoded_dir/error.log"
test ! -e "$encoded_dir/state%23prod.db"
