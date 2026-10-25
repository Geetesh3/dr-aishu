/* Dr. Aishu — search insights dashboard.
 *
 * Charts are hand-rolled SVG: no chart library, no CDN, so the page works
 * offline and in a Colab iframe with no third-party requests.
 *
 * The analytics endpoints are authenticated because aggregate medical data is
 * re-identifiable when a deployment has few users. The key is read from the same
 * sessionStorage slot the rest of the app uses (written by the developers page),
 * with an inline field as an override.
 */
(function () {
  "use strict";

  // Must match static/app.js and static/developers.js, or a key issued on the
  // developers page would never be found here.
  var KEY_SLOT = "draishu_dev_key";
  var $ = function (id) { return document.getElementById(id); };

  function storedKey() {
    try { return sessionStorage.getItem(KEY_SLOT) || ""; } catch (e) { return ""; }
  }

  function apiKey() {
    var typed = $("api-key") && $("api-key").value.trim();
    return typed || storedKey();
  }

  function setStatus(text, kind) {
    var el = $("status");
    el.textContent = text || "";
    el.className = "status" + (kind ? " " + kind : "");
  }

  async function api(path, options) {
    var opts = options || {};
    var headers = { "Content-Type": "application/json" };
    var key = apiKey();
    if (key) headers["X-API-Key"] = key;
    var res = await fetch(path, {
      method: opts.method || "GET",
      headers: headers,
      body: opts.body ? JSON.stringify(opts.body) : undefined
    });
    if (res.status === 401 || res.status === 403) {
      throw new Error("This dashboard needs an API key. Get a free one on the Developers page.");
    }
    if (!res.ok) {
      var detail = "";
      try { detail = (await res.json()).detail || ""; } catch (e) { /* non-JSON */ }
      throw new Error(detail || ("Request failed (" + res.status + ")"));
    }
    return res.json();
  }

  // ---- SVG helpers ---------------------------------------------------------
  var SVG = "http://www.w3.org/2000/svg";

  function el(name, attrs) {
    var node = document.createElementNS(SVG, name);
    for (var k in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) node.setAttribute(k, attrs[k]);
    }
    return node;
  }

  function emptyState(host, message) {
    host.innerHTML = "";
    var p = document.createElement("p");
    p.className = "empty";
    p.textContent = message;
    host.appendChild(p);
  }

  /** Horizontal bar chart for ranked categories. */
  function barChart(host, rows, opts) {
    opts = opts || {};
    if (!rows.length) return emptyState(host, opts.empty || "No data yet.");

    var rowH = 26, labelW = 150, valueW = 46, gap = 8;
    var width = 640;
    var barMax = width - labelW - valueW;
    var height = rows.length * rowH + gap;
    var max = Math.max.apply(null, rows.map(function (r) { return r.value; })) || 1;

    host.innerHTML = "";
    var svg = el("svg", { viewBox: "0 0 " + width + " " + height, role: "img" });

    rows.forEach(function (row, i) {
      var y = i * rowH;
      var w = Math.max(2, (row.value / max) * barMax);

      var label = el("text", { x: labelW - 10, y: y + rowH / 2 + 4, "text-anchor": "end", class: "bar-label" });
      label.textContent = row.label.length > 22 ? row.label.slice(0, 21) + "…" : row.label;
      svg.appendChild(label);

      svg.appendChild(el("rect", {
        x: labelW, y: y + 4, width: w, height: rowH - 12, rx: 4, fill: row.color || "#0f6b52"
      }));

      var value = el("text", { x: labelW + w + 8, y: y + rowH / 2 + 4, class: "bar-value" });
      value.textContent = row.display != null ? row.display : row.value;
      svg.appendChild(value);
    });

    host.appendChild(svg);
  }

  /** Filled area/line chart for the daily timeline. */
  function lineChart(host, points) {
    if (!points.length) return emptyState(host, "No data yet.");

    var width = 900, height = 200, padL = 34, padR = 10, padT = 12, padB = 26;
    var max = Math.max.apply(null, points.map(function (p) { return p.count; }));
    var scaleMax = Math.max(max, 1);
    var innerW = width - padL - padR, innerH = height - padT - padB;

    var x = function (i) {
      return points.length === 1 ? padL + innerW / 2 : padL + (i / (points.length - 1)) * innerW;
    };
    var y = function (v) { return padT + innerH - (v / scaleMax) * innerH; };

    host.innerHTML = "";
    var svg = el("svg", { viewBox: "0 0 " + width + " " + height, role: "img" });

    // horizontal gridlines with value labels
    [0, 0.5, 1].forEach(function (frac) {
      var gy = padT + innerH - frac * innerH;
      svg.appendChild(el("line", { x1: padL, y1: gy, x2: width - padR, y2: gy, class: "grid-line" }));
      var t = el("text", { x: padL - 6, y: gy + 4, "text-anchor": "end", class: "axis-label" });
      t.textContent = Math.round(frac * scaleMax);
      svg.appendChild(t);
    });

    var line = points.map(function (p, i) { return x(i) + "," + y(p.count); }).join(" ");
    svg.appendChild(el("polyline", {
      points: line, fill: "none", stroke: "#0f6b52", "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round"
    }));

    svg.appendChild(el("polygon", {
      points: padL + "," + (padT + innerH) + " " + line + " " + (width - padR) + "," + (padT + innerH),
      fill: "rgba(15,107,82,.10)"
    }));

    // label only first, middle and last dates so they never collide
    [0, Math.floor(points.length / 2), points.length - 1].forEach(function (i, n, arr) {
      if (arr.indexOf(i) !== n || !points[i]) return;
      var t = el("text", {
        x: x(i), y: height - 8, class: "axis-label",
        "text-anchor": i === 0 ? "start" : i === points.length - 1 ? "end" : "middle"
      });
      t.textContent = points[i].date.slice(5);
      svg.appendChild(t);
    });

    host.appendChild(svg);
  }

  var TRIAGE_COLOR = {
    emergency: "#b3261e",
    urgent: "#c2521a",
    "see-doctor": "#b4690e",
    routine: "#14794f",
    "self-care": "#0f6b52"
  };

  function statTile(value, label) {
    var d = document.createElement("div");
    d.className = "stat";
    var v = document.createElement("div");
    v.className = "value";
    v.textContent = value;
    var l = document.createElement("div");
    l.className = "label";
    l.textContent = label;
    d.appendChild(v); d.appendChild(l);
    return d;
  }

  function renderStats(totals, triageMix) {
    var host = $("stat-grid");
    host.innerHTML = "";
    host.appendChild(statTile(totals.searches.toLocaleString(), "Searches in window"));
    host.appendChild(statTile(totals.distinct_symptoms.toLocaleString(), "Distinct symptoms"));
    host.appendChild(statTile(totals.distinct_diagnoses.toLocaleString(), "Distinct diagnoses"));
    host.appendChild(statTile(
      totals.avg_confidence != null ? totals.avg_confidence + "%" : "—", "Avg. top-match confidence"));
    host.appendChild(statTile(
      totals.avg_duration_days != null ? totals.avg_duration_days + " d" : "—", "Avg. symptom duration"));
    host.appendChild(statTile(totals.with_image.toLocaleString(), "With an image"));
    host.appendChild(statTile(totals.consented_for_training.toLocaleString(), "Consented for training"));

    var urgent = triageMix
      .filter(function (t) { return t.triage === "emergency" || t.triage === "urgent"; })
      .reduce(function (a, t) { return a + t.count; }, 0);
    host.appendChild(statTile(urgent.toLocaleString(), "Emergency or urgent"));
  }

  function renderAll(data) {
    renderStats(data.totals, data.triage_mix);

    lineChart($("chart-timeline"), data.timeline);

    barChart($("chart-symptoms"), data.top_symptoms.map(function (s) {
      return { label: s.label, value: s.count };
    }), { empty: "No symptoms recorded yet." });

    barChart($("chart-diagnoses"), data.top_diagnoses.map(function (d) {
      return { label: d.label, value: d.count, display: d.count + " (" + d.share + "%)" };
    }), { empty: "No diagnoses recorded yet." });

    barChart($("chart-triage"), data.triage_mix.map(function (t) {
      return {
        label: t.triage, value: t.count,
        display: t.count + " (" + t.share + "%)",
        color: TRIAGE_COLOR[t.triage] || "#6d7c74"
      };
    }), { empty: "No triage data yet." });

    var sev = data.severity_distribution.filter(function (s) { return s.count > 0; });
    barChart($("chart-severity"), sev.map(function (s) {
      return {
        label: "Score " + s.score, value: s.count,
        color: s.score >= 8 ? "#b3261e" : s.score >= 5 ? "#c2521a" : "#14794f"
      };
    }), { empty: "No severity data yet." });
  }

  async function loadSummary() {
    var days = $("days").value;
    var includeDemo = $("include-demo").checked ? "true" : "false";
    var data = await api("/api/analytics/summary?days=" + days + "&include_demo=" + includeDemo);
    renderAll(data);
    return data;
  }

  async function loadBrands() {
    var data = await api("/api/analytics/brands?limit=15");
    $("brand-note").textContent = data.note || "";

    var host = $("brand-stats");
    host.innerHTML = "";
    host.appendChild(statTile(data.generics_covered.toLocaleString(), "Generics with brands"));
    host.appendChild(statTile(data.brands_total.toLocaleString(), "Brands listed"));
    host.appendChild(statTile(data.medicines_total.toLocaleString(), "Curated medicines"));

    var body = $("brand-table").querySelector("tbody");
    body.innerHTML = "";
    data.brands.forEach(function (b) {
      var tr = document.createElement("tr");
      var td1 = document.createElement("td"); td1.textContent = b.medicine;
      var td2 = document.createElement("td"); td2.className = "num"; td2.textContent = b.brands;
      var td3 = document.createElement("td"); td3.className = "num";
      td3.textContent = b.cheapest_inr != null ? "₹" + b.cheapest_inr : "—";
      tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
      body.appendChild(tr);
    });
  }

  /* ---------- real-world health data ---------- */

  // Sparkline for one indicator's series. Drawn as an SVG path; the axis is the
  // value range of the series itself, so the shape shows direction of travel.
  function sparkline(host, points) {
    if (!points || points.length < 2) return;
    var w = 240, h = 34, pad = 3;
    var values = points.map(function (p) { return p.value; });
    var min = Math.min.apply(null, values), max = Math.max.apply(null, values);
    var span = (max - min) || 1;
    var step = (w - pad * 2) / (points.length - 1);
    var d = points.map(function (p, i) {
      var x = pad + i * step;
      var y = h - pad - ((p.value - min) / span) * (h - pad * 2);
      return (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    }).join(" ");
    var svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("class", "spark");
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    svg.setAttribute("preserveAspectRatio", "none");
    var path = document.createElementNS(SVG, "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "1.5");
    svg.appendChild(path);
    host.appendChild(svg);
  }

  function formatValue(value) {
    if (value === null || value === undefined) return "—";
    if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(1) + "M";
    if (Math.abs(value) >= 1000) return Math.round(value).toLocaleString();
    return (Math.abs(value) < 10 ? value.toFixed(1) : String(Math.round(value * 10) / 10));
  }

  function renderWorld(data) {
    var grid = $("world-grid"), status = $("world-status"), src = $("world-sources");
    if (!grid) return;
    grid.innerHTML = "";
    if (!data || !data.available) {
      if (status) status.textContent = "No cached statistics for this country. "
        + "Run scripts/refresh_world_data.py on the host to fetch them.";
      if (src) src.textContent = "";
      return;
    }
    var shown = (data.indicators || []).filter(function (i) { return i.latest_value !== null && i.latest_value !== undefined; });
    if (status) {
      status.textContent = shown.length
        ? "Fetched " + (data.fetched_at || "").slice(0, 10) + " from the publishers' APIs. Year shown is the year the figure refers to."
        : "The publishers returned no values for this country.";
    }
    shown.forEach(function (ind) {
      var card = document.createElement("div");
      card.className = "world-card";
      var h = document.createElement("h4");
      h.textContent = ind.label;
      var v = document.createElement("div");
      v.className = "world-value";
      v.textContent = formatValue(ind.latest_value);
      var unit = document.createElement("small");
      unit.textContent = ind.unit;
      v.appendChild(unit);
      var meta = document.createElement("p");
      meta.textContent = ind.latest_year + " · " + ind.source;
      card.appendChild(h); card.appendChild(v); card.appendChild(meta);
      sparkline(card, ind.points);
      grid.appendChild(card);
    });
    if (src) {
      src.textContent = "Sources: " + (data.sources || []).map(function (s) {
        return s.name + " (" + s.license + ")";
      }).join(" · ");
    }
  }

  function loadWorld() {
    var sel = $("world-country");
    var country = sel ? sel.value : "IND";
    return fetch("/api/world-health?country=" + encodeURIComponent(country))
      .then(function (r) { return r.json(); })
      .then(renderWorld)
      .catch(function () { renderWorld(null); });
  }

  function withBusy(button, label, fn) {
    return async function () {
      var original = button.textContent;
      button.disabled = true;
      button.textContent = label;
      try {
        await fn();
      } catch (err) {
        setStatus(err.message, "error");
      } finally {
        button.disabled = false;
        button.textContent = original;
      }
    };
  }

  document.addEventListener("DOMContentLoaded", function () {
    // Prefill the inline field from the session key so the dashboard works
    // without retyping, while still allowing an override.
    var saved = storedKey();
    if (saved && $("api-key")) $("api-key").value = saved;

    $("refresh-btn").addEventListener("click", withBusy($("refresh-btn"), "Loading…", async function () {
      setStatus("Loading…");
      await loadSummary();
      await loadBrands();
      setStatus("Updated just now.", "ok");
    }));

    $("demo-btn").addEventListener("click", withBusy($("demo-btn"), "Generating…", async function () {
      setStatus("Running demo presentations through the analysis pipeline…");
      var res = await api("/api/analytics/demo", {
        method: "POST",
        body: { count: 250, days: parseInt($("days").value, 10) }
      });
      await loadSummary();
      setStatus("Generated " + res.created + " demo searches" +
        (res.failed ? " (" + res.failed + " skipped)" : "") + ".", "ok");
    }));

    $("purge-btn").addEventListener("click", withBusy($("purge-btn"), "Clearing…", async function () {
      var res = await api("/api/analytics/demo", { method: "DELETE" });
      await loadSummary();
      setStatus("Removed " + res.events_removed + " demo searches.", "ok");
    }));

    ["days", "include-demo"].forEach(function (id) {
      $(id).addEventListener("change", function () {
        loadSummary().catch(function (err) { setStatus(err.message, "error"); });
      });
    });

    var worldSel = $("world-country");
    if (worldSel) {
      worldSel.addEventListener("change", function () {
        loadWorld().catch(function () { renderWorld(null); });
      });
    }

    loadSummary()
      .then(loadBrands)
      .then(function () { setStatus(""); })
      .catch(function (err) { setStatus(err.message, "error"); });

    // Independent of the API key: these are public publisher statistics, so the
    // section still fills in for a visitor who has not signed in.
    loadWorld();
  });
})();
