/* Condition library page: search the ontology-backed condition library.
 *
 * Renders name, source, licence and identifier for every condition, because the
 * point of the library is that a name can be traced to the organisation that
 * published it. Results are paged from /api/conditions/search so the browser
 * never holds more than a page at a time -- the library has hundreds of
 * thousands of rows.
 */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const PAGE = 25;
  const SOURCE_LABEL = {
    mondo: "Mondo", doid: "Disease Ontology", orphanet: "Orphanet",
    icd10cm: "ICD-10-CM", mesh: "MeSH", kb: "Dr. Aishu knowledge base",
  };

  let offset = 0;
  let total = 0;
  let query = "";
  let filter = "all";
  let loading = false;

  const params = () => {
    const p = new URLSearchParams({ q: query, limit: String(PAGE), offset: String(offset) });
    if (filter === "assessable") p.set("assessable", "true");
    if (filter === "rare") p.set("rare", "true");
    return p;
  };

  const card = (c) => {
    const badges = [
      `<span class="lib-tag">${esc(SOURCE_LABEL[c.source] || c.source)}</span>`,
      c.has_details ? '<span class="lib-tag ok">Full assessment</span>' : "",
      c.is_rare ? '<span class="lib-tag warn">Rare</span>' : "",
      c.category ? `<span class="lib-tag">${esc(c.category)}</span>` : "",
    ].filter(Boolean).join("");
    return `
    <article class="lib-card">
      <header>
        <h3>${esc(c.name)}</h3>
        ${badges}
      </header>
      ${c.definition ? `<p>${esc(c.definition)}</p>` : ""}
      <p class="lib-meta">
        <code>${esc(c.id)}</code>
        ${c.license ? ` · ${esc(c.license)}` : ""}
        ${c.source_url ? ` · <a href="${esc(c.source_url)}" target="_blank" rel="noopener noreferrer">source</a>` : ""}
      </p>
    </article>`;
  };

  const render = (items, append) => {
    const el = $("#library-list");
    if (append) el.insertAdjacentHTML("beforeend", items.map(card).join(""));
    else el.innerHTML = items.length
      ? items.map(card).join("")
      : '<p class="muted">No conditions match that search.</p>';
    const count = $("#library-count");
    if (count) {
      count.textContent = total
        ? `Showing ${Math.min(offset + items.length, total).toLocaleString()} of ${total.toLocaleString()} conditions`
        : "No conditions match that search.";
    }
    const more = $("#library-more");
    if (more) more.hidden = offset + items.length >= total;
  };

  const fetchPage = async (append) => {
    if (loading) return;
    loading = true;
    const more = $("#library-more");
    if (more) more.textContent = "Loading…";
    try {
      const res = await fetch(`/api/conditions/search?${params()}`);
      const data = await res.json();
      total = data.total || 0;
      render(data.results || [], append);
    } catch {
      if (!append) $("#library-list").innerHTML = '<p class="muted">Could not load the library.</p>';
    } finally {
      loading = false;
      if (more) more.textContent = "Load more";
    }
  };

  const reload = () => { offset = 0; fetchPage(false); };

  const loadStats = async () => {
    try {
      const res = await fetch("/api/conditions/stats");
      const s = await res.json();
      const el = $("#library-stats");
      if (!s.available) {
        el.textContent = "The condition library has not been built on this host. "
          + "Run scripts/fetch_condition_sources.py then scripts/build_condition_library.py.";
        return;
      }
      el.innerHTML = `<strong>${(s.conditions || 0).toLocaleString()}</strong> conditions · `
        + `<strong>${(s.synonyms || 0).toLocaleString()}</strong> synonyms · `
        + `<strong>${(s.cross_references || 0).toLocaleString()}</strong> cross-references · `
        + `<strong>${(s.assessable || 0).toLocaleString()}</strong> with full assessment`;
      const grid = $("#library-sources");
      if (grid) {
        grid.innerHTML = (s.sources || []).map((src) => `
          <div class="source-card">
            <strong>${esc(SOURCE_LABEL[src.source] || src.source)}</strong>
            <span class="lib-tag">${esc(src.license)}</span>
            <p class="muted">${(src.count || 0).toLocaleString()} conditions</p>
          </div>`).join("");
      }
    } catch { /* stats are cosmetic; the search below still works */ }
  };

  const init = () => {
    loadStats();
    fetchPage(false);

    let timer = null;
    $("#library-search").addEventListener("input", (e) => {
      query = e.target.value;
      clearTimeout(timer);
      // Debounced so a fast typist does not fire a query per keystroke.
      timer = setTimeout(reload, 250);
    });

    $("#library-filters").addEventListener("click", (e) => {
      const btn = e.target.closest(".lib-filter");
      if (!btn) return;
      filter = btn.dataset.filter;
      document.querySelectorAll(".lib-filter").forEach((b) => b.classList.toggle("is-active", b === btn));
      reload();
    });

    $("#library-more").addEventListener("click", () => {
      offset += PAGE;
      fetchPage(true);
    });
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
