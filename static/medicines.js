/* Medicine catalogue page: curated medicines plus the full product catalogue. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  let medicines = [];
  let searchTimer = null;

  const curatedCard = (m) => `
    <article class="lib-card">
      <header>
        <h3>${esc(m.name)}</h3>
        <span class="lib-tag">${esc(m.drug_class)}</span>
        <span class="slot-badge ${m.otc ? "otc" : "rx"}">${m.otc ? "OTC" : "Rx"}</span>
      </header>
      <p class="muted">${esc((m.uses || []).join(" · "))}</p>
      <p><strong>Dose:</strong> ${esc(m.dose_adult)} — ${esc(m.frequency)}${m.with_food && m.with_food !== "any" ? `, ${esc(m.with_food)}` : ""}</p>
      ${m.max_daily_adult ? `<p><strong>Maximum daily:</strong> ${esc(m.max_daily_adult)}</p>` : ""}
      ${m.max_days ? `<p><strong>Maximum duration:</strong> ${esc(m.max_days)} days</p>` : ""}
      ${m.warnings && m.warnings.length ? `<p class="warn-text"><strong>Warnings:</strong> ${m.warnings.map(esc).join(" · ")}</p>` : ""}
    </article>`;

  const productCard = (p) => `
    <article class="lib-card product-card">
      <header>
        <h3>${esc(p.name)}</h3>
        ${p.combination ? '<span class="lib-tag warn">combination</span>' : ""}
        ${p.discontinued ? '<span class="lib-tag">discontinued</span>' : ""}
        ${p.curated_id ? '<span class="lib-tag ok">curated guidance</span>' : ""}
      </header>
      <p class="muted">${esc(p.molecule)}${p.form && p.form !== "unknown" ? ` · ${esc(p.form)}` : ""}${p.pack ? ` · ${esc(p.pack)}` : ""}</p>
      <p>${p.manufacturer ? esc(p.manufacturer) : ""}${p.price_inr ? ` — <strong>₹${esc(p.price_inr)}</strong>` : ""}</p>
    </article>`;

  const renderCurated = (query = "") => {
    const q = query.trim().toLowerCase();
    const list = !q ? medicines : medicines.filter((m) =>
      m.name.toLowerCase().includes(q) ||
      (m.generic || "").toLowerCase().includes(q) ||
      (m.drug_class || "").toLowerCase().includes(q) ||
      (m.uses || []).some((u) => u.toLowerCase().includes(q)));
    $("#med-count").textContent = `${list.length} of ${medicines.length} curated medicines`;
    $("#med-list").innerHTML = list.length
      ? list.map(curatedCard).join("")
      : '<p class="muted">No curated medicines match that search.</p>';
  };

  const searchCatalogue = async (query) => {
    const box = $("#catalogue-results");
    if (!box) return;
    if (query.trim().length < 2) {
      box.innerHTML = "";
      $("#catalogue-count").textContent = "";
      return;
    }
    box.innerHTML = '<p class="muted">Searching…</p>';
    try {
      const res = await fetch(`/api/products/search?q=${encodeURIComponent(query)}&limit=30`);
      const data = await res.json();
      $("#catalogue-count").textContent =
        `${data.count} of ${data.total_products.toLocaleString()} products across ${data.total_molecules.toLocaleString()} molecules`;
      box.innerHTML = data.results.length
        ? data.results.map(productCard).join("")
        : '<p class="muted">No products match that search.</p>';
    } catch {
      box.innerHTML = '<p class="muted">Could not search the catalogue.</p>';
    }
  };

  const init = async () => {
    try {
      const data = await (await fetch("/api/medicines")).json();
      medicines = data.medicines || [];
      $("#med-curated").textContent = medicines.length;
    } catch {
      $("#med-list").innerHTML = '<p class="muted">Could not load the catalogue.</p>';
      return;
    }

    try {
      const stats = await (await fetch("/api/products/stats")).json();
      $("#med-generics").textContent = (stats.molecules || 0).toLocaleString();
      $("#med-brands").textContent = (stats.products || 0).toLocaleString();
    } catch { /* counters stay as dashes */ }

    renderCurated();
    const input = $("#med-search");
    input.addEventListener("input", (e) => {
      const q = e.target.value;
      renderCurated(q);
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => searchCatalogue(q), 250);
    });
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
