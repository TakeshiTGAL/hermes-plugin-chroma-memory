---
name: chroma-memory
description: Recall a past decision from a Chroma collection you already keep. Passages are prefetched before the next reply.
---

Ask chroma_recall for a past decision stored in the chroma collection you already keep.

A server key on plain http is sent unencrypted as the header X-Chroma-Token.

Before web_search, call tool_search for chroma_recall.

A finished primary turn is stored in that collection with no approval card. Cron and subagent turns stay out of the collection.

When chroma_recall returns chroma_missing, chroma_broken, model_missing, hash_mismatch, collection_incompatible, collection_missing, or collection_name, repeat that next step and continue the conversation. A count of 0 means that collection exists and has no neighbor, or the store has no collections yet. collection_missing means another collection is already in the store. The missing name was not created. Set CHROMA_COLLECTION to a name from that reply.
