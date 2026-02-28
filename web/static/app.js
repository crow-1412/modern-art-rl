const state = {
  assets: { artists: {}, players: {} },
  game: null,
};

async function api(path, method = "GET", body = null) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function imageOrFallback(url, label) {
  if (url) return `<img src="${url}" alt="${label}" />`;
  return `<img alt="${label}" src="data:image/svg+xml;utf8,${encodeURIComponent(`<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'><rect width='100%' height='100%' fill='#33496b'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' fill='white' font-size='26'>${label}</text></svg>`)}"/>`;
}

function render() {
  if (!state.game) return;
  const { public: pub, seats, events } = state.game;

  const info = [
    ["Season", `${pub.season_idx + 1}/${pub.max_seasons}`],
    ["Turn", String(pub.turn_idx)],
    ["Auctioneer", String(pub.auctioneer)],
    ["Done", String(pub.done)],
  ];

  document.getElementById("publicInfo").innerHTML = info
    .map(([k, v]) => `<div class="kv"><div class="k">${k}</div><div>${v}</div></div>`)
    .join("");

  document.getElementById("artistBoard").innerHTML = pub.artist_order
    .map((artist) => {
      const img = state.assets.artists[artist];
      const played = pub.played_count[artist] ?? 0;
      const value = pub.value_block_sum[artist] ?? 0;
      return `<div class="artist">
        ${imageOrFallback(img, artist)}
        <div><b>${artist}</b></div>
        <div>Played: ${played}</div>
        <div>Value Blocks: ${value}</div>
      </div>`;
    })
    .join("");

  document.getElementById("players").innerHTML = Object.keys(seats)
    .map((pid) => {
      const s = seats[pid];
      const img = state.assets.players[pid];
      return `<div class="player">
        ${imageOrFallback(img, pid)}
        <div>
          <div><b>${pid}</b></div>
          <div>Cash: ${s.cash.toFixed(2)}</div>
          <div>Hand: ${s.hand_size}</div>
          <div>Owned this season: ${JSON.stringify(s.owned_this_season)}</div>
        </div>
      </div>`;
    })
    .join("");

  const lines = events.slice(-10).map((e, i) => `#${events.length - 10 + i + 1} T${e.turn_idx} S${e.season_idx}\n${JSON.stringify(e.info.lot_result, null, 2)}`);
  document.getElementById("eventLog").textContent = lines.join("\n\n");
}

async function refreshState() {
  state.game = await api("/api/state");
  render();
}

async function bootstrap() {
  state.assets = await api("/api/assets_manifest");
  state.game = await api("/api/new_game", "POST", { seed: 42 });
  render();

  document.getElementById("newGameBtn").onclick = async () => {
    state.game = await api("/api/new_game", "POST", { seed: Math.floor(Math.random() * 100000) });
    render();
  };

  document.getElementById("stepBtn").onclick = async () => {
    state.game = await api("/api/step", "POST");
    render();
  };

  document.getElementById("autoBtn").onclick = async () => {
    state.game = await api("/api/auto_play?max_steps=200", "POST");
    render();
  };
}

bootstrap().catch((e) => {
  document.getElementById("eventLog").textContent = `Error: ${e.message}`;
});
