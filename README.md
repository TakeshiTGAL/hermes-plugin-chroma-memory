# Chroma Memory

"what did we decide about authentication" → nearest passages from the Chroma collection you already keep.

"what should I remember from the chroma collection" → the same collection, pulled in before the next reply.

A collection with a different embedding function → that collection stays as it is, and the reply is the next step.

## Thirty seconds

```bash
if hermes plugins install --help 2>&1 | grep -q -- '--yes-deps'
then
  hermes plugins install "file://${PWD}" --enable --yes-deps
else
  hermes plugins install "file://${PWD}" --enable
fi
hermes plugins enable chroma-memory
hermes memory setup chroma-memory
hermes chat -q "What did we decide about authentication?"
```

The last line runs one turn with the model you already set up in Hermes. Prefetch runs before that turn.

The block reads `hermes plugins install --help`. When that text contains `--yes-deps`, the install uses `--enable --yes-deps`. Otherwise the install is `hermes plugins install --enable`. Hermes 0.21.4 takes the second command.

On Hermes 0.22.2, `hermes plugins install --enable --yes-deps` sets `memory.provider` to `chroma-memory`. On Hermes 0.21.4, `hermes plugins install --enable` writes `plugins.enabled` only, and prefetch does not start. `hermes memory setup chroma-memory` writes `memory.provider` on 0.21.4 and on 0.22.2.

An empty profile has no embedding archive yet. The tool returns `model_missing` and names the pinned file, the size, and the sha256. A full agent turn is what stores the exchange. The archive has to be in the profile before that store can embed the turn.

## Details

Enabling the plugin registers `chroma_recall`. Prefetch runs when `memory.provider` is `chroma-memory`.

The default store is an embedded Chroma directory at `plugin-data/chroma-memory/store` under this profile. Set `CHROMA_PATH` to an absolute path when the collection already lives somewhere else, including outside the profile. A relative path is refused. A symlink at the final path component is refused. The archive and `config.json` stay under `plugin-data/chroma-memory` even when `CHROMA_PATH` points outside.

Set `CHROMA_SERVER_URL` to an `http` or `https` origin with no user name, password, path, or query. The default port for `http` here is 8000. The default port for `https` is 443. A key for that server is sent only as the header `X-Chroma-Token`. Over plain http, that key is sent unencrypted.

Chroma Cloud is used when the server URL is empty and `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE` are all set. Those three are sent only to `api.trychroma.com` on port 443. A partial Cloud trio returns `cloud_config` and writes nothing. A server URL wins over Cloud fields.

Every one of those values is read with `get_secret` on each call: `CHROMA_SERVER_URL`, `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE`, `CHROMA_COLLECTION`, `CHROMA_PATH`, `CHROMA_MODEL_DOWNLOAD`. The collection name defaults to `hermes`. `hermes memory setup` can store `collection` and `path` in `plugin-data/chroma-memory/config.json`. Secret fields are dropped from that file.

A read does not create a collection. A write creates the named collection only when that store has no collections yet. When other collections are already there and the requested name is absent, the reply is `collection_missing` and it lists the names that are there. Nothing is written. An empty collection that does exist returns `ok` true and `count` 0.

Chroma 1.5.9 accepts a collection name of 3 to 512 characters from letters, digits, dot, underscore, and hyphen. The name starts and ends with a letter or digit. Two periods in a row are refused. An IPv4 address is refused. `José` and `a/b?"` stop at `collection_name`. A query can contain those characters. `999.999.999.999` is a legal name. `1.2.3.4` is refused.

The embedding archive is `onnx.tar.gz`, 83178821 bytes, sha256 `913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3`, from `https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz`. A mismatch is refused and the archive is left unextracted. The fetch uses an empty proxy. Server and Cloud HTTP follow the proxy of the Chroma client in this process. A stuck server call stays a daemon thread until the process exits. It is one thread.

`chroma_recall` fetches that archive only when `CHROMA_MODEL_DOWNLOAD` is `1`, `true`, or `yes`. Prefetch and turn storage do not start that fetch. Place the file yourself, or set the flag and call the tool.

Finished primary turns are stored with no approval card. Yolo and `approvals.mode: off` do not add a card. Cron and subagent turns are skipped. A missing `agent_context` counts as primary. Empty user text and empty assistant text are skipped. Anyone who can talk to the agent can call `chroma_recall`.

`hermes tools disable chroma-memory` hides the tool on that surface. Prefetch keeps running while `memory.provider` is `chroma-memory`. `hermes memory off` clears that key and prefetch stops. `hermes plugins remove chroma-memory` clears it too. Remove leaves `plugin-data/chroma-memory`, including the store, the archive, and `config.json`.

A checkout of this plugin, `.git` included, is 372 kilobytes (`du -sk`).

A missing or broken Chroma returns a next step and the conversation continues. A collection whose stored embedding-function name is outside `default` and `onnx_mini_lm_l6_v2` is left unread. A legacy collection with no stored name and dimension 384 is read as all-MiniLM-L6-v2. Chroma 1.5.9 keeps unregistered functions as `legacy` and does not keep a separate archive id. The declared dependency is `chromadb>=1.5.9,<1.6`. This plugin starts no MCP process.

The distance number in a passage is the number Chroma returns.
