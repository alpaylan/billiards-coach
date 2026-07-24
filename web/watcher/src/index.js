// bilardo.info match watcher — a Cloudflare Worker.
//
// Cron (every 5 min): polls https://bilardo.info/json.php?type=3cushion (and
// pool), keeps the last state in KV, and pushes a notification on transitions
// (no matches -> live matches, and table lineup changes) to an ntfy.sh topic
// and/or a custom webhook. HTTP:
//   GET /         current status (live check, plus last-seen state)
//   GET /history  recent transition events (newest first)
//
// Config (wrangler.toml vars / secrets):
//   NTFY_TOPIC   ntfy.sh topic to push to ("" disables)
//   WEBHOOK_URL  optional: POST full JSON transition events here (secret)
//   GH_TOKEN     optional secret: fine-grained PAT (Contents: read+write on
//                GH_REPO) — when set, a matches-started transition fires a
//                repository_dispatch (type "bilardo-match") so the track-match
//                workflow runs. The payload carries the raw match objects; the
//                workflow proceeds only if it finds a video_url in them.
//   GH_REPO      "owner/repo" for the dispatch (var)

const TYPES = ["3cushion", "pool"];

async function poll(type) {
  const url = `https://bilardo.info/json.php?type=${type}&t=${Math.floor(Date.now() / 1000)}`;
  const r = await fetch(url, { headers: { "user-agent": "billiards-coach-watcher" } });
  if (!r.ok) throw new Error(`${type}: HTTP ${r.status}`);
  const j = await r.json();
  // Field names verified against a live tournament (2026-07-23): players are
  // player1name/player2name, and `broadcast` carries the YouTube video id of
  // the table's stream (broadcast_status: "live"/"offline") — the key that
  // makes automatic corpus capture possible.
  const matches = (j.matches || []).map((m) => ({
    table: m.table_no ?? null,
    players: [m.player1name ?? m.player1 ?? m.p1 ?? null, m.player2name ?? m.player2 ?? m.p2 ?? null],
    video: m.broadcast ? `https://www.youtube.com/watch?v=${m.broadcast}` : null,
    live: m.broadcast_status === "live",
    raw: m,
  }));
  return { type, count: j.matchcount ?? matches.length, matches };
}

// A compact fingerprint of "what's on": type:count:tables. Player names change
// per match — notify on lineup changes, not on every score tick.
function fingerprint(states) {
  return states
    .map((s) => `${s.type}:${s.count}:${s.matches.map((m) => m.table).join("+")}`)
    .join("|");
}

// Returns per-channel delivery results (e.g. {ntfy: 200}) so a silent failure
// (ntfy rate-limits shared Cloudflare egress IPs with 429s) shows up in the
// transition history instead of vanishing.
async function notify(env, title, body) {
  const jobs = [];
  if (env.NTFY_TOPIC) {
    jobs.push(
      fetch(`https://ntfy.sh/${env.NTFY_TOPIC}`, {
        method: "POST",
        headers: { Title: title, Priority: "high", Tags: "8ball" },
        body,
      }).then((r) => ["ntfy", r.status], (e) => ["ntfy", String(e)])
    );
  }
  if (env.WEBHOOK_URL) {
    jobs.push(
      fetch(env.WEBHOOK_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ title, body, at: new Date().toISOString() }),
      }).then((r) => ["webhook", r.status], (e) => ["webhook", String(e)])
    );
  }
  const results = await Promise.all(jobs);
  return Object.fromEntries(results);
}

async function check(env) {
  const states = [];
  for (const t of TYPES) {
    try {
      states.push(await poll(t));
    } catch (e) {
      states.push({ type: t, count: null, error: String(e), matches: [] });
    }
  }
  const now = new Date().toISOString();
  const fp = fingerprint(states);
  const prev = (await env.STATE.get("last", "json")) || {};
  const total = states.reduce((n, s) => n + (s.count || 0), 0);

  if (fp !== prev.fp) {
    const prevTotal = prev.total || 0;
    let title = null;
    if (total > 0 && prevTotal === 0) title = `bilardo.info: ${total} live match${total > 1 ? "es" : ""} started`;
    else if (total === 0 && prevTotal > 0) title = "bilardo.info: streams ended";
    else if (total > 0) title = `bilardo.info: lineup changed (${total} live)`;
    if (title) {
      const lines = states
        .flatMap((s) =>
          s.matches.map(
            (m) =>
              `[${s.type}] M${m.table}: ${m.players.filter(Boolean).join(" vs ")}${m.video ? ` ${m.video}` : ""}`
          )
        )
        .join("\n");
      const delivered = await notify(env, title, lines || "no details");
      // Matches just started: kick the cloud tracker (dormant until GH_TOKEN set).
      let dispatched = null;
      if (total > 0 && prevTotal === 0 && env.GH_TOKEN && env.GH_REPO) {
        // Per-table stream links, live tables first — the workflow picks from
        // `videos`; `video_url` (first live stream) kept for compatibility.
        const videos = states
          .flatMap((s) =>
            s.matches
              .filter((m) => m.video)
              .map((m) => ({ type: s.type, table: m.table, players: m.players, url: m.video, live: m.live }))
          )
          .sort((a, b) => Number(b.live) - Number(a.live));
        const video_url = videos[0]?.url || null;
        dispatched = await fetch(`https://api.github.com/repos/${env.GH_REPO}/dispatches`, {
          method: "POST",
          headers: {
            authorization: `Bearer ${env.GH_TOKEN}`,
            accept: "application/vnd.github+json",
            "user-agent": "billiards-coach-watcher",
          },
          body: JSON.stringify({
            event_type: "bilardo-match",
            // preset: these tournament streams carry the overhead inset on the
            // LEFT (bilardo_l in build_match.py's PRESETS).
            client_payload: {
              at: now,
              total,
              video_url,
              videos,
              preset: "bilardo_l",
              // Unique per dispatch: auto-runs previously all defaulted to
              // name "match" and collided on artifacts/checkpoints.
              name: `auto_${now.slice(0, 10)}_${now.slice(11, 16).replace(":", "")}`,
              states,
            },
          }),
        }).then((r) => r.status, (e) => String(e));
      }
      const history = (await env.STATE.get("history", "json")) || [];
      history.unshift({ at: now, title, total, fp, delivered, dispatched });
      await env.STATE.put("history", JSON.stringify(history.slice(0, 100)));
    }
  }
  await env.STATE.put("last", JSON.stringify({ at: now, fp, total, states }));

  // PERMANENT MATCH REGISTRY: every broadcast id ever seen, keyed by video id.
  // The YouTube links vanish from the API when a match ends (and the streams
  // are often unlisted afterward) — a link not recorded while live is a match
  // lost to the corpus. Upsert on every poll; GET /matches serves the archive.
  const live = states.flatMap((s) =>
    s.matches
      .filter((m) => m.raw?.broadcast)
      .map((m) => ({
        id: m.raw.broadcast,
        type: s.type,
        table: m.table,
        players: m.players,
        status: m.raw.broadcast_status ?? null,
        scores: [m.raw.player1score ?? null, m.raw.player2score ?? null],
      }))
  );
  if (live.length) {
    const seen = (await env.STATE.get("matches", "json")) || {};
    for (const m of live) {
      // One broadcast id = one TABLE's daily stream; matches rotate within it
      // (observed live: same id, new player pair at 0-0). Track player-pair
      // SESSIONS with first/last-seen times — a per-match segmentation index
      // for the day's video, for free.
      const sess = { players: m.players, scores: m.scores, status: m.status, first_seen: now, last_seen: now };
      const prev = seen[m.id];
      if (!prev) {
        seen[m.id] = {
          id: m.id,
          type: m.type,
          table: m.table,
          url: `https://www.youtube.com/watch?v=${m.id}`,
          first_seen: now,
          last_seen: now,
          sessions: [sess],
        };
      } else {
        prev.last_seen = now;
        if (!prev.sessions) {
          // migrate a pre-sessions entry
          prev.sessions = [{ players: prev.players, scores: prev.scores, status: prev.status, first_seen: prev.first_seen, last_seen: prev.first_seen }];
          delete prev.players;
          delete prev.scores;
          delete prev.status;
        }
        const cur = prev.sessions[prev.sessions.length - 1];
        if (cur && JSON.stringify(cur.players) === JSON.stringify(m.players)) {
          cur.last_seen = now;
          cur.scores = m.scores;
          cur.status = m.status;
        } else {
          prev.sessions.push(sess);
        }
      }
    }
    await env.STATE.put("matches", JSON.stringify(seen));
  }
  return { at: now, total, states };
}

export default {
  async scheduled(_event, env, ctx) {
    ctx.waitUntil(check(env));
  },
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname === "/history") {
      const history = (await env.STATE.get("history", "json")) || [];
      return Response.json(history);
    }
    if (url.pathname === "/matches") {
      // The permanent registry: every match (broadcast id) ever witnessed.
      const seen = (await env.STATE.get("matches", "json")) || {};
      const list = Object.values(seen).sort((a, b) => (a.first_seen < b.first_seen ? 1 : -1));
      return Response.json({ count: list.length, matches: list });
    }
    // live check on demand; falls back to last stored state on upstream error
    let current;
    try {
      current = await check(env);
    } catch {
      current = await env.STATE.get("last", "json");
    }
    return Response.json(current);
  },
};
