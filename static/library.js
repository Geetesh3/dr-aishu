/* Condition library page: fetch the knowledge base and render searchable cards. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  let diseases = [];

  const card = (d) => `
    <article class="lib-card">
      <header>
        <h3>${esc(d.name)}</h3>
        <span class="lib-tag">${esc(d.category)}</span>
        ${d.contagious ? '<span class="lib-tag warn">contagious</span>' : ""}
      </header>
      <p class="muted">${esc(d.description)}</p>
      <p><strong>Symptoms:</strong> ${(d.symptoms || []).map(esc).join(", ") || "—"}</p>
      ${d.self_care && d.self_care.length ? `<p><strong>Self-care:</strong> ${d.self_care.map(esc).join(" · ")}</p>` : ""}
      ${d.when_to_see_doctor && d.when_to_see_doctor.length
        ? `<p class="warn-text"><strong>See a doctor if:</strong> ${d.when_to_see_doctor.map(esc).join(" · ")}</p>` : ""}
    </article>`;

  const render = (query = "") => {
    const q = query.trim().toLowerCase();
    const list = !q ? diseases : diseases.filter((d) =>
      d.name.toLowerCase().includes(q) ||
      d.category.toLowerCase().includes(q) ||
      (d.description || "").toLowerCase().includes(q) ||
      (d.symptoms || []).some((s) => s.toLowerCase().includes(q)));
    const el = $("#library-list");
    const count = $("#library-count");
    if (count) count.textContent = `${list.length} of ${diseases.length} conditions`;
    el.innerHTML = list.length ? list.map(card).join("") : '<p class="muted">No conditions match that search.</p>';
  };

  const init = async () => {
    try {
      const res = await fetch("/api/diseases");
      diseases = (await res.json()).diseases || [];
    } catch {
      $("#library-list").innerHTML = '<p class="muted">Could not load the library.</p>';
      return;
    }
    render();
    $("#library-search").addEventListener("input", (e) => render(e.target.value));
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
