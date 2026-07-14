"use strict";
/*
 * mcp-memory knowledge-graph viewer.
 *
 * A dependency-free, Neo4j-Browser-style graph explorer rendered on a <canvas>:
 * force-directed layout, captioned circular nodes coloured by entity type,
 * curved multi-edges with relationship-type pills + arrowheads, a node
 * inspector, a live legend, paginated browse + full-text search, and — the
 * headline interaction — double-click a node to expand its relationships.
 *
 * Data endpoints (same server that serves /mcp; no MCP tools involved):
 *   GET /ui/graph?entityType&offset&limit → a page of the graph
 *   GET /ui/search?q&entityType&offset&limit → a page of FTS matches
 *   GET /ui/expand?name&depth&direction   → a node's neighbourhood
 * All return {entities, relations, entityTypes, stats, page:{offset,limit,returned,hasMore}}.
 */
(function () {
  const $ = (id) => document.getElementById(id);
  const cv = $("cv"), ctx = cv.getContext("2d");
  const SEP = "\u0000"; // key delimiter — a NUL byte never appears in real names

  // Neo4j Browser's default categorical label palette. Types are assigned a
  // colour in first-seen order and it sticks for the session.
  const PALETTE = [
    "#FFDF81", "#C990C0", "#F79767", "#57C7E3", "#F16667", "#D9C8AE",
    "#8DCC93", "#ECB5C9", "#4C8EDA", "#FFC454", "#DA7194", "#569480",
    "#848484", "#B2B2B2", "#B0C4DE", "#B58AA5",
  ];
  const colorOf = new Map();
  let paletteNext = 0;
  function colorForType(t) {
    if (!colorOf.has(t)) { colorOf.set(t, PALETTE[paletteNext % PALETTE.length]); paletteNext++; }
    return colorOf.get(t);
  }

  // ── Auth token: URL hash (#token=…, never sent to the server / not logged),
  //    else sessionStorage. Kept client-side; forwarded as a Bearer header.
  function readHashToken() {
    const m = /[#&]token=([^&]+)/.exec(location.hash || "");
    return m ? decodeURIComponent(m[1]) : null;
  }
  let token = readHashToken() || sessionStorage.getItem("mcpmem_token") || "";
  if (readHashToken()) {
    sessionStorage.setItem("mcpmem_token", token);
    history.replaceState(null, "", location.pathname + location.search);
  }

  // ── State ─────────────────────────────────────────────────────────────────
  const view = { x: 0, y: 0, k: 1 };
  const browse = { offset: 0, limit: 300, query: "", entityType: "" }; // paged browse/search cursor
  let page = { offset: 0, limit: 300, returned: 0, hasMore: false };
  let nodes = [], links = [], nodeById = new Map();
  let totalStats = null;
  let selected = null, hover = null, pinnedDrag = null;
  let alpha = 0, raf = null, busyReq = false;
  let dpr = Math.max(1, window.devicePixelRatio || 1);
  // Cap on nodes held in the canvas at once. Browse pages are already bounded
  // (≤1000, server-enforced); this bounds *expansion* so double-clicking a hub
  // can't push the force layout into tens of thousands of nodes.
  const MAX_RENDER_NODES = 3000;

  const toScreen = (wx, wy) => ({ x: (wx + view.x) * view.k, y: (wy + view.y) * view.k });
  const toWorld = (sx, sy) => ({ x: sx / view.k - view.x, y: sy / view.k - view.y });

  function resize() {
    dpr = Math.max(1, window.devicePixelRatio || 1);
    const r = cv.getBoundingClientRect();
    cv.width = Math.round(r.width * dpr);
    cv.height = Math.round(r.height * dpr);
    requestDraw();
  }
  window.addEventListener("resize", resize);

  // ── API & overlay ──────────────────────────────────────────────────────────
  function api(path) {
    const headers = {};
    if (token) headers["Authorization"] = "Bearer " + token;
    return fetch(path, { headers });
  }
  function overlay(title, msg, opts = {}) {
    $("ovTitle").textContent = title;
    $("ovMsg").textContent = msg || "";
    $("ovTokRow").style.display = opts.token ? "flex" : "none";
    $("overlay").classList.toggle("err", !!opts.err);
    $("overlay").classList.add("show");
    if (opts.token) $("ovToken").focus();
  }
  const hideOverlay = () => $("overlay").classList.remove("show");
  async function handleError(res) {
    if (res.ok) return false;
    if (res.status === 401) overlay("Authentication required", "This server requires a bearer token.", { token: true, err: true });
    else if (res.status === 403) overlay("Graph reading disabled", (await res.text().catch(() => "")) || "Start the server with --enable-graph-read (or --enable-all).", { err: true });
    else overlay("Error " + res.status, await res.text().catch(() => ""), { err: true });
    return true;
  }

  // ── Paginated load (browse overview OR full-text search) ───────────────────
  function setBusy(b) {
    busyReq = b;
    for (const id of ["searchBtn", "overview", "prev", "next"]) $(id).disabled = b;
    if (!b) updatePager(); // restore correct prev/next disabled state
  }
  async function load() {
    if (busyReq) return;
    setBusy(true);
    overlay("Loading…", browse.query ? `Searching for “${browse.query}”…` : "Fetching the knowledge graph.");
    const p = new URLSearchParams();
    if (browse.entityType) p.set("entityType", browse.entityType);
    p.set("offset", String(browse.offset));
    p.set("limit", String(browse.limit));
    const path = browse.query ? "/ui/search?q=" + encodeURIComponent(browse.query) + "&" + p : "/ui/graph?" + p;
    let res;
    try { res = await api(path); }
    catch (e) { overlay("Connection failed", String(e), { err: true }); setBusy(false); return; }
    if (await handleError(res)) { setBusy(false); return; }
    const data = await res.json();
    setBusy(false);
    setGraph(data);
  }

  function runSearch() {
    browse.query = $("search").value.trim();
    browse.offset = 0;
    load();
  }
  function showOverview() {
    $("search").value = "";
    browse.query = "";
    browse.offset = 0;
    load();
  }
  function gotoPage(delta) {
    const next = browse.offset + delta * browse.limit;
    if (next < 0 || (delta > 0 && !page.hasMore)) return;
    browse.offset = Math.max(0, next);
    load();
  }

  // ── Expand (double-click traversal) ────────────────────────────────────────
  async function expand(node) {
    if (!node || node._loading) return;
    node._loading = true; kick();
    let res;
    try { res = await api("/ui/expand?depth=1&direction=both&name=" + encodeURIComponent(node.id)); }
    catch (e) { node._loading = false; flash("expand failed: " + e); return; }
    node._loading = false;
    if (!res.ok) { flash("expand failed (" + res.status + ")"); if (res.status === 401 || res.status === 403) await handleError(res); kick(); return; }
    const data = await res.json();
    node.expanded = true;
    const { added, capped } = mergeGraph(data, node);
    if (capped) flash(`node limit ${fmt(MAX_RENDER_NODES)} reached — dismiss or isolate to explore further`);
    else flash(added ? `+${added} node${added === 1 ? "" : "s"}` : "no new relationships");
  }

  // ── Graph (re)building ─────────────────────────────────────────────────────
  function makeNode(e, x, y) {
    const hasObs = Array.isArray(e.observations);
    return {
      id: e.name, type: e.entityType || "",
      // The browse/search list payloads omit observation *bodies* (they carry
      // only `obsCount`); the inspector lazy-loads bodies via /ui/node on select.
      // `obs === null` means "bodies not loaded yet"; `obsCount` is always known.
      obs: hasObs ? e.observations : null,
      obsCount: hasObs ? e.observations.length : (e.obsCount | 0),
      color: colorForType(e.entityType || ""), // cache: avoid a Map lookup per node per frame
      x, y, vx: 0, vy: 0, deg: 0, fixed: false, expanded: false, _loading: false,
      _lblR: -1, _lbl: "", // cached fitted label + the screen-radius bucket it was measured at
    };
  }
  function recomputeDegrees() {
    for (const n of nodes) n.deg = 0;
    for (const l of links) { l.source.deg++; l.target.deg++; }
    assignEdgeSlots();
  }
  // Fan parallel/reciprocal edges out as separate curved arcs.
  function assignEdgeSlots() {
    const groups = new Map();
    for (const l of links) {
      const key = l.source.id < l.target.id ? l.source.id + SEP + l.target.id : l.target.id + SEP + l.source.id;
      let g = groups.get(key); if (!g) { g = []; groups.set(key, g); }
      l._group = g; g.push(l);
    }
    for (const g of groups.values()) g.forEach((l, i) => { l._slot = i - (g.length - 1) / 2; });
  }

  // Even, overlap-free seed positions (sunflower/phyllotaxis spiral) for a fresh
  // page. Gives the layout real structure on the first frame so we don't need an
  // expensive synchronous pre-roll before fitting — the animation settles from a
  // sane start instead of from random noise.
  function spiralPos(i) {
    const a = i * 2.399963229728653; // golden angle
    const r = SPRING_LEN * 0.9 * Math.sqrt(i + 0.5);
    return { x: Math.cos(a) * r, y: Math.sin(a) * r };
  }
  function setGraph(data) {
    page = data.page || { offset: browse.offset, limit: browse.limit, returned: (data.entities || []).length, hasMore: false };
    const prev = new Map(nodes.map((n) => [n.id, n]));
    let seed = 0;
    nodes = (data.entities || []).map((e) => {
      const p = prev.get(e.name);
      const s = p ? null : spiralPos(seed++);
      const n = makeNode(e, p ? p.x : s.x, p ? p.y : s.y);
      if (p) n.fixed = p.fixed;
      return n;
    });
    nodeById = new Map(nodes.map((n) => [n.id, n]));
    links = (data.relations || [])
      .filter((r) => nodeById.has(r.from) && nodeById.has(r.to))
      .map((r) => ({ source: nodeById.get(r.from), target: nodeById.get(r.to), type: r.relationType || "" }));
    recomputeDegrees();

    for (const t of (data.entityTypes || [])) colorForType(t.type);
    fillTypeFilter(data.entityTypes || []);
    totalStats = data.stats || totalStats;

    selectNode(null);
    updateStats(); updatePager();
    if (!nodes.length) {
      overlay(browse.query ? "No matches" : "Empty graph",
        browse.query ? `Nothing matched “${browse.query}”.` : "No entities to display for this filter.");
    } else hideOverlay();
    alpha = 1; fitView(true); kick();
  }

  // Merge an expansion result; returns the number of newly added nodes.
  function mergeGraph(data, origin) {
    let added = 0, capped = false;
    for (const e of (data.entities || [])) {
      if (nodeById.has(e.name)) continue;
      if (nodes.length >= MAX_RENDER_NODES) { capped = true; break; } // bound the layout
      const ang = Math.random() * Math.PI * 2, rad = 60 + Math.random() * 40;
      const n = makeNode(e, origin.x + Math.cos(ang) * rad, origin.y + Math.sin(ang) * rad);
      nodes.push(n); nodeById.set(n.id, n); added++;
    }
    const seen = new Set(links.map((l) => l.source.id + SEP + l.type + SEP + l.target.id));
    for (const r of (data.relations || [])) {
      const s = nodeById.get(r.from), t = nodeById.get(r.to);
      if (!s || !t) continue; // an endpoint was past the node cap — skip the edge
      const key = r.from + SEP + (r.relationType || "") + SEP + r.to;
      if (seen.has(key)) continue;
      seen.add(key);
      links.push({ source: s, target: t, type: r.relationType || "" });
    }
    recomputeDegrees(); buildLegend(); updateStats();
    if (selected === origin) selectNode(origin); // refresh inspector relation list
    alpha = Math.max(alpha, 0.45); kick(); // gentle reheat — keep existing layout calm
    return { added, capped };
  }

  function dismiss(node) {
    nodes = nodes.filter((n) => n !== node);
    links = links.filter((l) => l.source !== node && l.target !== node);
    nodeById.delete(node.id);
    if (selected === node) selectNode(null);
    recomputeDegrees(); buildLegend(); updateStats(); kick();
  }
  function isolate(node) {
    const keep = new Set([node]);
    for (const l of links) { if (l.source === node) keep.add(l.target); else if (l.target === node) keep.add(l.source); }
    nodes = nodes.filter((n) => keep.has(n));
    nodeById = new Map(nodes.map((n) => [n.id, n]));
    links = links.filter((l) => keep.has(l.source) && keep.has(l.target));
    recomputeDegrees(); buildLegend(); updateStats(); fitView(false); kick();
  }

  // ── Toolbar / legend / stats / pager ───────────────────────────────────────
  const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : n);
  function updateStats() {
    const e = totalStats ? totalStats.entities : nodes.length;
    const r = totalStats ? totalStats.relations : links.length;
    $("stats").textContent = `${fmt(nodes.length)} / ${fmt(e)} nodes · ${fmt(links.length)} / ${fmt(r)} rels`;
    buildLegend();
  }
  function updatePager() {
    const from = page.returned ? page.offset + 1 : 0;
    const to = page.offset + page.returned;
    $("pageLabel").textContent = page.returned ? `${fmt(from)}–${fmt(to)}` : "0";
    $("prev").disabled = busyReq || page.offset === 0;
    $("next").disabled = busyReq || !page.hasMore;
  }
  function fillTypeFilter(types) {
    const sel = $("typeFilter"), cur = sel.value;
    sel.innerHTML = '<option value="">all labels</option>';
    for (const t of types) {
      const o = document.createElement("option");
      o.value = t.type; o.textContent = `${t.type} (${t.count})`;
      sel.appendChild(o);
    }
    sel.value = cur;
  }
  function buildLegend() {
    const counts = new Map();
    for (const n of nodes) counts.set(n.type, (counts.get(n.type) || 0) + 1);
    const el = $("legend"); el.innerHTML = "";
    [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 14).forEach(([type, c]) => {
      const d = document.createElement("span");
      d.className = "chip";
      d.innerHTML = `<span class="sw" style="background:${colorForType(type)}"></span>${escapeHtml(type || "—")} <span class="n">${c}</span>`;
      el.appendChild(d);
    });
  }
  let flashTimer = null;
  function flash(msg) {
    $("stats").textContent = msg;
    clearTimeout(flashTimer);
    flashTimer = setTimeout(updateStats, 1500);
  }

  // ── Force simulation ───────────────────────────────────────────────────────
  const REPULSION = 9000, SPRING = 0.02, SPRING_LEN = 130, GRAVITY = 0.012, DAMP = 0.9;
  const radius = (n) => 15 + Math.min(24, Math.sqrt(n.deg) * 4);
  const anyLoading = () => nodes.some((n) => n._loading);

  // ── Barnes-Hut quadtree: O(n log n) repulsion instead of O(n²) all-pairs, so
  //    large graphs (and hub expansions) stay at interactive frame rates.
  const THETA2 = 0.81; // (θ=0.9)² — cells smaller than θ·distance are lumped into one body
  function insert(q, n, depth) {
    q.cx = (q.cx * q.mass + n.x) / (q.mass + 1);
    q.cy = (q.cy * q.mass + n.y) / (q.mass + 1);
    q.mass++;
    if (q.mass === 1) { q.node = n; return; }
    if (q.size < 1e-3 || depth > 48) return; // coincident cluster — stop subdividing
    if (!q.quads) { q.quads = [null, null, null, null]; const old = q.node; q.node = null; if (old) place(q, old, depth); }
    place(q, n, depth);
  }
  function place(q, n, depth) {
    const half = q.size / 2;
    const i = (n.x >= q.x + half ? 1 : 0) + (n.y >= q.y + half ? 2 : 0);
    let c = q.quads[i];
    if (!c) c = q.quads[i] = { x: q.x + (i & 1 ? half : 0), y: q.y + (i & 2 ? half : 0), size: half, cx: 0, cy: 0, mass: 0, node: null, quads: null };
    insert(c, n, depth + 1);
  }
  function buildTree(ns) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of ns) { if (n.x < minX) minX = n.x; if (n.y < minY) minY = n.y; if (n.x > maxX) maxX = n.x; if (n.y > maxY) maxY = n.y; }
    if (!isFinite(minX)) return null;
    const size = Math.max(maxX - minX, maxY - minY, 1);
    const root = { x: minX, y: minY, size, cx: 0, cy: 0, mass: 0, node: null, quads: null };
    for (const n of ns) insert(root, n, 0);
    return root;
  }
  function repulse(q, n, alpha) {
    if (!q || q.mass === 0) return;
    let dx = n.x - q.cx, dy = n.y - q.cy, d2 = dx * dx + dy * dy;
    if (q.quads && q.size * q.size >= THETA2 * d2) { // cell too close to approximate → recurse
      for (const c of q.quads) if (c) repulse(c, n, alpha);
      return;
    }
    if (q.node === n && q.mass === 1) return; // the node's own leaf
    if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 1; }
    const f = (REPULSION * q.mass / d2) * alpha, inv = 1 / Math.sqrt(d2);
    n.vx += dx * inv * f; n.vy += dy * inv * f;
  }

  function step() {
    if (alpha < 0.02) return false;
    const n = nodes.length;
    if (n > 1) {
      const tree = buildTree(nodes);
      for (let i = 0; i < n; i++) repulse(tree, nodes[i], alpha);
    }
    for (const l of links) {
      const dx = l.target.x - l.source.x, dy = l.target.y - l.source.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = SPRING * (dist - SPRING_LEN) * alpha;
      const fx = (dx / dist) * f, fy = (dy / dist) * f;
      l.source.vx += fx; l.source.vy += fy; l.target.vx -= fx; l.target.vy -= fy;
    }
    for (const a of nodes) {
      a.vx -= a.x * GRAVITY * alpha; a.vy -= a.y * GRAVITY * alpha;
      if (a.fixed || a === pinnedDrag) { a.vx = 0; a.vy = 0; continue; }
      a.vx *= DAMP; a.vy *= DAMP; a.x += a.vx; a.y += a.vy;
    }
    alpha *= 0.985;
    return true;
  }
  function kick() { alpha = Math.max(alpha, 0.55); if (!raf) loop(); }
  function loop() {
    const busy = step();
    draw();
    raf = (busy || pinnedDrag || anyLoading()) ? requestAnimationFrame(loop) : null;
  }
  // Coalesce one-off redraws (hover, pan, zoom, selection) into a single rAF so
  // several events in one frame don't each paint. A no-op while the sim loop is
  // already running (it paints every frame).
  let drawPending = false;
  function requestDraw() {
    if (raf || drawPending) return;
    drawPending = true;
    requestAnimationFrame(() => { drawPending = false; draw(); });
  }

  // ── Rendering ──────────────────────────────────────────────────────────────
  const searchTerm = () => ($("search").value || "").trim().toLowerCase();
  const matches = (n, q) => q && (n.id.toLowerCase().includes(q) || n.type.toLowerCase().includes(q));
  function neighborsOf(node) {
    const s = new Set();
    for (const l of links) { if (l.source === node) s.add(l.target); else if (l.target === node) s.add(l.source); }
    return s;
  }
  function edgeControl(l) {
    const a = l.source, b = l.target;
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    if (!l._slot) return { x: mx, y: my };
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
    const off = l._slot * 34;
    return { x: mx + (-dy / d) * off, y: my + (dx / d) * off };
  }

  function draw() {
    ctx.save();
    ctx.scale(dpr, dpr);
    const W = cv.width / dpr, H = cv.height / dpr;
    ctx.clearRect(0, 0, W, H);

    const q = searchTerm();
    const focus = selected || hover;
    const nbrs = focus ? neighborsOf(focus) : null;
    const showRelLabels = view.k > 0.75;
    // Viewport-cull margin — generous enough to cover node radius, curved edges
    // and relationship pills that bow just outside the strict box.
    const M = 80;

    for (const l of links) {
      const a = toScreen(l.source.x, l.source.y), b = toScreen(l.target.x, l.target.y);
      // Skip edges wholly off one side of the viewport (both endpoints outside).
      if ((a.x < -M && b.x < -M) || (a.x > W + M && b.x > W + M) ||
          (a.y < -M && b.y < -M) || (a.y > H + M && b.y > H + M)) continue;
      const ec = edgeControl(l), c = toScreen(ec.x, ec.y);
      const active = focus && (l.source === focus || l.target === focus);
      ctx.lineWidth = active ? 2 : 1.2;
      ctx.strokeStyle = active ? "rgba(1,139,255,.85)" : (focus ? "rgba(150,158,168,.22)" : "rgba(150,158,168,.55)");
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(c.x, c.y, b.x, b.y); ctx.stroke();
      drawArrow(c, b, radius(l.target) * view.k, active);
      if (l.type && (active || showRelLabels)) {
        const mx = 0.25 * a.x + 0.5 * c.x + 0.25 * b.x, my = 0.25 * a.y + 0.5 * c.y + 0.25 * b.y;
        drawRelLabel(mx, my, l.type, active);
      }
    }

    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const n of nodes) {
      const s = toScreen(n.x, n.y), r = radius(n) * view.k;
      if (s.x < -M || s.x > W + M || s.y < -M || s.y > H + M) continue; // off-screen
      const dim = focus && n !== focus && !(nbrs && nbrs.has(n));
      ctx.globalAlpha = dim ? 0.3 : 1;
      ctx.beginPath(); ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
      ctx.fillStyle = n.color; ctx.fill();
      if (n === selected) { ctx.lineWidth = 3; ctx.strokeStyle = "rgba(1,139,255,.9)"; ctx.stroke(); }
      else if (q && matches(n, q)) { ctx.lineWidth = 3; ctx.strokeStyle = "#f0a020"; ctx.stroke(); }
      else { ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(0,0,0,.12)"; ctx.stroke(); }
      if (n.expanded) { ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(0,0,0,.28)"; ctx.beginPath(); ctx.arc(s.x, s.y, r + 3, 0, Math.PI * 2); ctx.stroke(); }
      if (n._loading) drawSpinner(s.x, s.y, r + 7);
      if (!dim && r > 13) {
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#2a2c34";
        ctx.font = `${Math.max(9, Math.min(13, r * 0.5))}px -apple-system, system-ui, sans-serif`;
        // Cache the fitted (truncated) label per integer screen-radius. fit()'s
        // measureText is a canvas hotspot; the screen radius is constant frame to
        // frame while the layout settles (it only changes on zoom or degree),
        // so this reduces measureText from every-node-every-frame to near-zero.
        const rb = r | 0;
        if (n._lblR !== rb) { n._lbl = fit(ctx, n.id, r * 1.8); n._lblR = rb; }
        ctx.fillText(n._lbl, s.x, s.y);
      }
    }
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  function fit(c, text, maxW) {
    if (c.measureText(text).width <= maxW) return text;
    let lo = 0, hi = text.length;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (c.measureText(text.slice(0, mid) + "…").width <= maxW) lo = mid; else hi = mid - 1; }
    return lo > 0 ? text.slice(0, lo) + "…" : "";
  }
  function drawArrow(from, to, targetR, active) {
    const dx = to.x - from.x, dy = to.y - from.y, d = Math.hypot(dx, dy) || 1;
    const ux = dx / d, uy = dy / d;
    const tipX = to.x - ux * (targetR + 1.5), tipY = to.y - uy * (targetR + 1.5), sz = active ? 9 : 7;
    ctx.fillStyle = active ? "rgba(1,139,255,.85)" : "rgba(150,158,168,.75)";
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - ux * sz - uy * sz * 0.5, tipY - uy * sz + ux * sz * 0.5);
    ctx.lineTo(tipX - ux * sz + uy * sz * 0.5, tipY - uy * sz - ux * sz * 0.5);
    ctx.closePath(); ctx.fill();
  }
  function drawRelLabel(x, y, text, active) {
    ctx.font = "10px -apple-system, system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = active ? "rgba(1,139,255,.95)" : "rgba(255,255,255,.9)";
    roundRect(x - w / 2, y - 8, w, 16, 8); ctx.fill();
    if (!active) { ctx.strokeStyle = "rgba(150,158,168,.5)"; ctx.lineWidth = 1; roundRect(x - w / 2, y - 8, w, 16, 8); ctx.stroke(); }
    ctx.fillStyle = active ? "#fff" : "#5a616e";
    ctx.fillText(text, x, y);
  }
  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  let spinPhase = 0;
  function drawSpinner(x, y, r) {
    spinPhase += 0.3;
    ctx.strokeStyle = "rgba(1,139,255,.9)"; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(x, y, r, spinPhase, spinPhase + Math.PI * 1.4); ctx.stroke();
  }

  // ── Hit testing & interaction ──────────────────────────────────────────────
  function nodeAt(sx, sy) {
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i], s = toScreen(n.x, n.y), r = radius(n) * view.k + 2;
      if ((sx - s.x) ** 2 + (sy - s.y) ** 2 <= r * r) return n;
    }
    return null;
  }
  let dragging = false, dragMoved = false, last = { x: 0, y: 0 };
  cv.addEventListener("mousedown", (e) => {
    const rect = cv.getBoundingClientRect();
    const n = nodeAt(e.clientX - rect.left, e.clientY - rect.top);
    dragging = true; dragMoved = false; last = { x: e.clientX, y: e.clientY };
    pinnedDrag = n || null; cv.classList.add("grabbing");
  });
  window.addEventListener("mousemove", (e) => {
    const rect = cv.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    if (dragging) {
      const dx = e.clientX - last.x, dy = e.clientY - last.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) dragMoved = true;
      last = { x: e.clientX, y: e.clientY };
      if (pinnedDrag) {
        const w = toWorld(sx, sy);
        pinnedDrag.x = w.x; pinnedDrag.y = w.y; pinnedDrag.vx = 0; pinnedDrag.vy = 0;
        if (!raf) loop(); // the sim loop paints each frame while a node is pinned
      } else { view.x += dx / view.k; view.y += dy / view.k; requestDraw(); }
      return;
    }
    const n = nodeAt(sx, sy);
    if (n !== hover) { hover = n; requestDraw(); }
    const tt = $("tooltip");
    if (n) {
      const oc = n.obsCount | 0;
      tt.style.display = "block";
      tt.style.left = Math.min(sx + 14, rect.width - 300) + "px";
      tt.style.top = (sy + 16) + "px";
      const o = oc ? `<div class="tt-sub">${fmt(oc)} observation${oc > 1 ? "s" : ""} · double-click to expand</div>` : `<div class="tt-sub">double-click to expand</div>`;
      tt.innerHTML = `<div class="tt-name">${escapeHtml(n.id)}</div><div class="tt-sub">${escapeHtml(n.type || "—")} · degree ${n.deg}</div>${o}`;
      cv.style.cursor = "pointer";
    } else { tt.style.display = "none"; cv.style.cursor = dragging ? "grabbing" : "grab"; }
  });
  window.addEventListener("mouseup", () => {
    if (dragging && pinnedDrag && !dragMoved) selectNode(pinnedDrag);
    else if (dragging && pinnedDrag && dragMoved) pinnedDrag.fixed = true;
    else if (dragging && !pinnedDrag && !dragMoved) selectNode(null);
    dragging = false; pinnedDrag = null; cv.classList.remove("grabbing");
  });
  cv.addEventListener("dblclick", (e) => {
    const rect = cv.getBoundingClientRect();
    const n = nodeAt(e.clientX - rect.left, e.clientY - rect.top);
    if (n) { expand(n); selectNode(n); }
  });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = cv.getBoundingClientRect();
    zoomAt(e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.0015));
  }, { passive: false });

  function zoomAt(sx, sy, factor) {
    const before = toWorld(sx, sy);
    view.k = Math.min(4, Math.max(0.05, view.k * factor));
    const after = toWorld(sx, sy);
    view.x += after.x - before.x; view.y += after.y - before.y;
    requestDraw();
  }
  const zoomCenter = (f) => zoomAt(cv.width / dpr / 2, cv.height / dpr / 2, f);

  // ── Inspector ──────────────────────────────────────────────────────────────
  function selectNode(n) {
    selected = n;
    const ins = $("inspector");
    if (!n) { ins.classList.remove("show"); $("zoom").classList.remove("shift"); requestDraw(); return; }
    $("insName").textContent = n.id;
    $("insDot").style.background = n.color;
    const rels = [];
    for (const l of links) {
      if (l.source === n) rels.push({ dir: "out", other: l.target.id, type: l.type });
      else if (l.target === n) rels.push({ dir: "in", other: l.source.id, type: l.type });
    }
    const relHtml = rels.length ? rels.map((r) =>
      r.dir === "out"
        ? `<div class="rel"><span class="rt">${escapeHtml(r.type)}</span><span class="arrow">→</span><a data-goto="${escapeHtml(r.other)}">${escapeHtml(r.other)}</a></div>`
        : `<div class="rel dir-in"><a data-goto="${escapeHtml(r.other)}">${escapeHtml(r.other)}</a><span class="rt">${escapeHtml(r.type)}</span><span class="arrow">→</span></div>`
    ).join("") : `<div class="meta">No relations loaded — double-click to expand.</div>`;
    // Observation bodies are lazy-loaded (list payloads carry only obsCount).
    const oc = n.obsCount | 0;
    let obsHtml;
    if (n.obs === null && oc > 0) {
      obsHtml = `<div class="meta">Loading ${fmt(oc)} observation${oc === 1 ? "" : "s"}…</div>`;
      loadObservations(n);
    } else {
      const obs = n.obs || [];
      obsHtml = obs.length ? obs.map((o) => `<div class="obs">${escapeHtml(o)}</div>`).join("") : `<div class="meta">No observations.</div>`;
    }
    $("insBody").innerHTML =
      `<span class="pill" style="background:${n.color}">${escapeHtml(n.type || "—")}</span>` +
      `<div class="meta">degree ${n.deg} · ${fmt(oc)} observation${oc === 1 ? "" : "s"}</div>` +
      `<div class="sec">Observations</div>${obsHtml}` +
      `<div class="sec">Relationships (${rels.length})</div>${relHtml}`;
    ins.classList.add("show"); $("zoom").classList.add("shift");
    $("insBody").querySelectorAll("[data-goto]").forEach((a) =>
      a.addEventListener("click", () => { const t = nodeById.get(a.getAttribute("data-goto")); if (t) { centerOn(t); selectNode(t); } }));
    requestDraw();
  }
  // Lazy-fetch observation bodies for the inspected node; re-render if it's still
  // selected when they arrive. On failure, mark as loaded-empty so we don't retry.
  async function loadObservations(n) {
    try {
      const res = await api("/ui/node?name=" + encodeURIComponent(n.id));
      if (!res.ok) { n.obs = []; if (selected === n) selectNode(n); return; }
      const data = await res.json();
      n.obs = data.observations || [];
      n.obsCount = n.obs.length;
    } catch { n.obs = []; }
    if (selected === n) selectNode(n);
  }
  function centerOn(n) {
    const W = cv.width / dpr, H = cv.height / dpr;
    view.x = W / (2 * view.k) - n.x; view.y = H / (2 * view.k) - n.y;
    requestDraw();
  }
  function fitView(reset) {
    if (!nodes.length) { view.x = 0; view.y = 0; view.k = 1; requestDraw(); return; }
    // A short Barnes-Hut warmup relaxes the (already spiral-seeded) layout enough
    // to fit sensibly, without the old 80-iteration synchronous O(n²) pre-roll
    // that froze the main thread on every load. The rAF loop finishes settling.
    if (reset) {
      const a = alpha; alpha = 1;
      const iters = nodes.length > 400 ? 12 : 30;
      for (let i = 0; i < iters; i++) step();
      alpha = a;
    }
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of nodes) { minX = Math.min(minX, n.x); minY = Math.min(minY, n.y); maxX = Math.max(maxX, n.x); maxY = Math.max(maxY, n.y); }
    const W = cv.width / dpr, H = cv.height / dpr, pad = 110;
    const gw = Math.max(1, maxX - minX), gh = Math.max(1, maxY - minY);
    view.k = Math.min(2.2, Math.max(0.05, Math.min((W - pad) / gw, (H - pad) / gh)));
    view.x = W / (2 * view.k) - (minX + maxX) / 2;
    view.y = H / (2 * view.k) - (minY + maxY) / 2;
    requestDraw();
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ── Controls ────────────────────────────────────────────────────────────────
  $("searchBtn").addEventListener("click", runSearch);
  $("overview").addEventListener("click", showOverview);
  $("typeFilter").addEventListener("change", () => { browse.entityType = $("typeFilter").value; browse.offset = 0; load(); });
  $("limit").addEventListener("change", () => { const v = parseInt($("limit").value, 10); if (v > 0) { browse.limit = Math.min(1000, v); browse.offset = 0; load(); } });
  $("search").addEventListener("input", requestDraw); // live-highlight loaded nodes as you type
  $("search").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
  $("prev").addEventListener("click", () => gotoPage(-1));
  $("next").addEventListener("click", () => gotoPage(1));
  $("zoomIn").addEventListener("click", () => zoomCenter(1.3));
  $("zoomOut").addEventListener("click", () => zoomCenter(1 / 1.3));
  $("zoomFit").addEventListener("click", () => fitView(false));
  $("insClose").addEventListener("click", () => selectNode(null));
  $("insExpand").addEventListener("click", () => selected && expand(selected));
  $("insIsolate").addEventListener("click", () => selected && isolate(selected));
  $("insDismiss").addEventListener("click", () => selected && dismiss(selected));
  $("ovGo").addEventListener("click", () => { token = $("ovToken").value.trim(); if (token) sessionStorage.setItem("mcpmem_token", token); load(); });
  $("ovToken").addEventListener("keydown", (e) => { if (e.key === "Enter") $("ovGo").click(); });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { if (selected) selectNode(null); }
  });

  resize();
  load();
})();
