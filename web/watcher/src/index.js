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

const TYPES = ["3cushion", "pool"];

async function poll(type) {
  const url = `https://bilardo.info/json.php?type=${type}&t=${Math.floor(Date.now() / 1000)}`;
  const r = await fetch(url, { headers: { "user-agent": "billiards-coach-watcher" } });
  if (!r.ok) throw new Error(`${type}: HTTP ${r.status}`);
  const j = await r.json();
  const matches = (j.matches || []).map((m) => ({
    table: m.table_no ?? null,
    players: [m.player1 ?? m.p1 ?? null, m.player2 ?? m.p2 ?? null],
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

async function notify(env, title, body) {
  const jobs = [];
  if (env.NTFY_TOPIC) {
    jobs.push(
      fetch(`https://ntfy.sh/${env.NTFY_TOPIC}`, {
        method: "POST",
        headers: { Title: title, Priority: "high", Tags: "8ball" },
        body,
      })
    );
  }
  if (env.WEBHOOK_URL) {
    jobs.push(
      fetch(env.WEBHOOK_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ title, body, at: new Date().toISOString() }),
      })
    );
  }
  await Promise.allSettled(jobs);
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
        .flatMap((s) => s.matches.map((m) => `[${s.type}] M${m.table}: ${m.players.filter(Boolean).join(" vs ")}`))
        .join("\n");
      await notify(env, title, lines || "no details");
      const history = (await env.STATE.get("history", "json")) || [];
      history.unshift({ at: now, title, total, fp });
      await env.STATE.put("history", JSON.stringify(history.slice(0, 100)));
    }
  }
  await env.STATE.put("last", JSON.stringify({ at: now, fp, total, states }));
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
