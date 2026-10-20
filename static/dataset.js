/* Dr. Aishu — operator dataset browser.
 *
 * Read-only view over the built Excel workbook. Every endpoint it calls needs an
 * operator (Pro) key: the workbook is external published data, but the response
 * also carries build internals such as file paths and archive hashes, so the
 * view is not public. The key comes from the same sessionStorage slot the rest
 * of the app uses (written by the developers page), with the gate field as an
 * override.
 *
 * Every value is inserted with textContent, never innerHTML, because the strings
 * come from third-party datasets and a condition name is not a place to trust.
 */
(function () {
  "use strict";

  var KEY_SLOT = "draishu_dev_key";
  var $ = function (id) { return document.getElementById(id); };

  function storedKey() {
    try { return sessionStorage.getItem(KEY_SLOT) || ""; } catch (e) { return ""; }
  }

  function saveKey(value) {
    try { sessionStorage.setItem(KEY_SLOT, value); } catch (e) { /* private mode */ }
  }

  function forgetKey() {
    try { sessionStorage.removeItem(KEY_SLOT); } catch (e) { /* private mode */ }
  }

  function currentKey() {
    var typed = $("ds-key") && $("ds-key").value.trim();
    return typed || storedKey();
  }

  function setStatus(text, kind) {
    var node = $("ds-status");
    if (!node) return;
    node.textContent = text || "";
    node.className = "status" + (kind ? " is-" + kind : "");
  }

  function setGateNote(text, kind) {
    var node = $("ds-gate-note");
    if (!node) return;
    node.textContent = text || "";
    node.className = "muted small" + (kind ? " is-" + kind : "");
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function fmtBytes(n) {
    if (!n) return "—";
    var units = ["B", "kB", "MB", "GB", "TB"];
    var i = 0;
    var v = Number(n);
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return (i === 0 ? v : v.toFixed(1)) + " " + units[i];
  }

  async function api(path) {
    var headers = { "Accept": "application/json" };
    var key = currentKey();
    if (key) headers["X-API-Key"] = key;
    var res = await fetch(path, { headers: headers });
    if (!res.ok) {
      var detail = "";
      try { detail = (await res.json()).detail || ""; } catch (e) { /* non-JSON */ }
      var err = new Error(detail || ("Request failed (" + res.status + ")"));
      err.status = res.status;
      throw err;
    }
    return res.json();
  }

  // ---- stat tiles ----------------------------------------------------------
  var SHEET_LABELS = {
    Diseases: "Conditions",
    Symptoms: "Symptoms",
    Disease_Symptoms: "Disease–symptom pairs",
    Precautions: "Precaution sets",
    Images: "Images",
    MedicineNet: "MedicineNet index",
    Sources: "Sources"
  };

  function renderStats(status) {
    var grid = $("ds-stats");
    grid.textContent = "";
    var sheets = status.sheets || {};
    Object.keys(SHEET_LABELS).forEach(function (name) {
      if (sheets[name] === undefined) return;
      var card = el("article", "stat");
      card.appendChild(el("span", "stat-value", sheets[name].toLocaleString()));
      card.appendChild(el("span", "stat-label", SHEET_LABELS[name]));
      grid.appendChild(card);
    });
    if (status.bytes) {
      var card = el("article", "stat");
      card.appendChild(el("span", "stat-value", fmtBytes(status.bytes)));
      card.appendChild(el("span", "stat-label", "Workbook size"));
      grid.appendChild(card);
    }
  }

  // ---- sources -------------------------------------------------------------
  var LICENSE_NOTE = {
    "unknown": "No licence declared — all rights reserved by default",
    "none-declared": "No licence declared — all rights reserved by default",
    "cc-by-nc-4.0": "Non-commercial use only",
    "copyright-webmd": "Copyright WebMD — index only"
  };

  function renderSources(rows) {
    var body = $("ds-sources").querySelector("tbody");
    body.textContent = "";
    rows.forEach(function (s) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", null, s.title || s.id));

      var lic = el("td");
      lic.appendChild(el("code", null, s.license || "unknown"));
      if (LICENSE_NOTE[s.license]) {
        lic.appendChild(el("div", "muted", LICENSE_NOTE[s.license]));
      }
      tr.appendChild(lic);

      tr.appendChild(el("td", null, s.status || ""));
      tr.appendChild(el("td", "num", fmtBytes(s.bytes)));

      var sha = el("td");
      sha.appendChild(el("code", null, (s.sha256 || "—").slice(0, 16) + (s.sha256 ? "…" : "")));
      if (s.url) {
        var a = el("a", null, "source");
        a.href = s.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        sha.appendChild(document.createTextNode(" "));
        sha.appendChild(a);
      }
      tr.appendChild(sha);
      body.appendChild(tr);
    });
  }

  // ---- trained image model -------------------------------------------------
  function renderModel(info) {
    var body = $("ds-model").querySelector("tbody");
    var note = $("ds-model-note");
    body.textContent = "";
    if (!info || !info.available) {
      note.textContent = (info && info.hint) || "No image model trained yet.";
      return;
    }
    var mods = info.modalities || {};
    Object.keys(mods).forEach(function (name) {
      var m = mods[name];
      var tr = document.createElement("tr");
      tr.appendChild(el("td", null, name));
      tr.appendChild(el("td", null, (m.classes || []).length + " — " + (m.classes || []).join(", ")));
      tr.appendChild(el("td", "num", m.accuracy == null ? "—" : (m.accuracy * 100).toFixed(1) + "%"));
      tr.appendChild(el("td", "num",
        m.cv_accuracy == null ? "—"
          : (m.cv_accuracy * 100).toFixed(1) + "% ± " + ((m.cv_std || 0) * 100).toFixed(1)));
      body.appendChild(tr);
    });
    note.textContent = "Trained " + (info.trained_at || "—") +
      ". Trained on " + Object.keys(mods).map(function (k) {
        return (mods[k].n_train || 0).toLocaleString() + " " + k;
      }).join(" and ") + " images. Split: " +
      ((mods[Object.keys(mods)[0]] || {}).split || "group") + ".";
  }

  // ---- images --------------------------------------------------------------
  function renderImages(labels) {
    var grid = $("ds-image-labels");
    grid.textContent = "";
    Object.keys(labels || {}).forEach(function (modality) {
      var classes = labels[modality];
      var total = Object.keys(classes).reduce(function (n, k) { return n + classes[k]; }, 0);
      var card = el("article", "stat");
      card.appendChild(el("span", "stat-value", total.toLocaleString()));
      card.appendChild(el("span", "stat-label", modality + " images"));
      var detail = Object.keys(classes).sort().map(function (k) {
        return k + " (" + classes[k].toLocaleString() + ")";
      }).join(", ");
      card.appendChild(el("span", "muted", detail));
      grid.appendChild(card);
    });
  }

  // ---- conditions ----------------------------------------------------------
  var diseaseCache = null;

  function renderDiseases(rows, total) {
    var body = $("ds-diseases").querySelector("tbody");
    body.textContent = "";
    rows.forEach(function (d) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", null, d.name));
      tr.appendChild(el("td", null, (d.in_kb || "").toLowerCase() === "yes" ? "yes" : "—"));
      tr.appendChild(el("td", null, d.category || "—"));
      tr.appendChild(el("td", null, (d.treatments || "—").toString().slice(0, 90)));
      tr.appendChild(el("td", "muted", d.sources || ""));
      body.appendChild(tr);
    });
    fillSummary($("ds-disease-count"), [
      ["Showing", rows.length],
      ["of", total],
      ["conditions", null],
    ]);
  }

  /* A summary bar is a row of "label value" pairs; a null value is a bare
   * label, which keeps the wording flexible without string concatenation. */
  function fillSummary(node, pairs) {
    node.textContent = "";
    pairs.forEach(function (pair) {
      var span = document.createElement("span");
      if (pair[1] === null || pair[1] === undefined) {
        span.textContent = pair[0];
      } else {
        span.appendChild(document.createTextNode(pair[0] + " "));
        span.appendChild(el("strong", null, pair[1].toLocaleString()));
      }
      node.appendChild(span);
    });
  }

  function applyDiseaseFilter() {
    if (!diseaseCache) return;
    var onlyKb = $("ds-in-kb").checked;
    var q = ($("ds-search").value || "").trim().toLowerCase();
    var rows = diseaseCache.filter(function (d) {
      if (onlyKb && (d.in_kb || "").toLowerCase() !== "yes") return false;
      if (!q) return true;
      return (d.name || "").toLowerCase().indexOf(q) !== -1;
    });
    renderDiseases(rows.slice(0, 100), rows.length);
  }

  // ---- MedicineNet ---------------------------------------------------------
  function renderMedicineNet(rows, total) {
    var body = $("ds-medicinenet").querySelector("tbody");
    body.textContent = "";
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", null, r.condition));
      tr.appendChild(el("td", null, r.letter || ""));
      var cell = el("td");
      if (r.url) {
        var a = el("a", null, "open");
        a.href = r.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        cell.appendChild(a);
      }
      tr.appendChild(cell);
      body.appendChild(tr);
    });
    fillSummary($("ds-mn-count"), [
      ["Showing", rows.length],
      ["of", total],
      ["index entries", null],
    ]);
  }

  // ---- DeepLesion ----------------------------------------------------------
  function renderDeepLesion(rows, total, types) {
    var body = $("ds-deeplesion").querySelector("tbody");
    body.textContent = "";
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", null, r.file_name || ""));
      tr.appendChild(el("td", "num", r.coarse_lesion_type || ""));
      tr.appendChild(el("td", null, r.patient_sex || ""));
      tr.appendChild(el("td", "num", r.patient_age || ""));
      tr.appendChild(el("td", "num", r.split || ""));
      tr.appendChild(el("td", "muted", (r.bounding_boxes || "").slice(0, 40)));
      body.appendChild(tr);
    });
    fillSummary($("ds-dl-count"), [
      ["Showing", rows.length],
      ["of", total],
      ["lesion annotations", null],
    ]);
    var select = $("ds-lesion-type");
    if (select && select.options.length <= 1) {
      Object.keys(types || {}).forEach(function (code) {
        var opt = document.createElement("option");
        opt.value = code;
        opt.textContent = "Type " + code + " (" + types[code].toLocaleString() + ")";
        select.appendChild(opt);
      });
    }
  }

  // ---- public dataset catalogue -------------------------------------------
  function renderCatalogue(rows) {
    var body = $("ds-catalogue").querySelector("tbody");
    body.textContent = "";
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", null, r.disease || ""));
      tr.appendChild(el("td", null, r.dataset || ""));
      var cell = el("td");
      if (r.url) {
        var a = el("a", null, "open");
        a.href = r.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        cell.appendChild(a);
      }
      tr.appendChild(cell);
      body.appendChild(tr);
    });
    fillSummary($("ds-cat-count"), [
      ["Showing", rows.length],
      ["public datasets", null],
    ]);
  }

  // ---- load ----------------------------------------------------------------
  async function loadAll() {
    var status = await api("/api/dataset/status");
    if (!status.available) {
      setStatus(
        "The dataset workbook is not built yet. Run: python scripts/fetch_sources.py " +
        "and then python scripts/build_dataset_workbook.py",
        "error"
      );
      return;
    }
    renderStats(status);
    renderModel(status.image_model);

    var sources = await api("/api/dataset/sources");
    renderSources(sources.items || []);

    var images = await api("/api/dataset/images?limit=1");
    renderImages(images.labels);

    var diseases = await api("/api/dataset/diseases?limit=0");
    diseaseCache = diseases.items || [];
    renderDiseases(diseaseCache.slice(0, 100), diseaseCache.length);

    var mn = await api("/api/dataset/medicinenet?limit=100");
    renderMedicineNet(mn.items || [], mn.total);

    var dl = await api("/api/dataset/deeplesion?limit=100");
    renderDeepLesion(dl.items || [], dl.total, dl.lesion_types || {});

    var cat = await api("/api/dataset/catalogue");
    renderCatalogue(cat.items || []);
  }

  function showContent() {
    $("ds-gate").hidden = true;
    $("ds-content").hidden = false;
  }

  function showGate(message, kind) {
    $("ds-gate").hidden = false;
    $("ds-content").hidden = true;
    if (message) setGateNote(message, kind);
  }

  async function unlock() {
    var key = currentKey();
    if (!key) {
      setGateNote("Enter an operator key to continue.", "error");
      return;
    }
    setGateNote("Checking key…");
    setStatus("");
    try {
      await loadAll();
      saveKey(key);
      showContent();
      setStatus("");
      setGateNote("");
    } catch (err) {
      if (err.status === 401) {
        forgetKey();
        showGate("That key was rejected. Check it and try again.", "error");
      } else if (err.status === 403) {
        showGate("That key does not have the operator scope. Use a Pro key.", "error");
      } else {
        showGate(err.message, "error");
      }
    }
  }

  async function init() {
    var saved = storedKey();
    if (!saved) {
      showGate("This page needs an operator key.");
      return;
    }
    if ($("ds-key")) $("ds-key").value = saved;
    try {
      await loadAll();
      showContent();
    } catch (err) {
      if (err.status === 401 || err.status === 403) forgetKey();
      showGate(err.status === 403
        ? "The saved key does not have the operator scope. Use a Pro key."
        : "The saved key was rejected. Enter a valid operator key.",
        "error");
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    var connect = $("ds-connect");
    var forget = $("ds-forget");
    var input = $("ds-key");
    var kb = $("ds-in-kb");
    var search = $("ds-search");
    if (connect) connect.addEventListener("click", unlock);
    if (input) input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") unlock();
    });
    if (forget) forget.addEventListener("click", function () {
      forgetKey();
      if (input) input.value = "";
      showGate("Signed out.");
    });
    if (kb) kb.addEventListener("change", applyDiseaseFilter);
    if (search) search.addEventListener("input", applyDiseaseFilter);
    var lesionType = $("ds-lesion-type");
    if (lesionType) {
      lesionType.addEventListener("change", async function () {
        var code = lesionType.value;
        var path = "/api/dataset/deeplesion?limit=100";
        if (code) path += "&lesion_type=" + encodeURIComponent(code);
        var body = await api(path);
        renderDeepLesion(body.items || [], body.total, body.lesion_types || {});
      });
    }
    init();
  });
})();
