/* Home page: knowledge-base counters.
 *
 * The language chooser lives in the site header only. The home page used to
 * also render a chip grid under "Sections"; two controls for one setting meant
 * two places to keep in sync and two things to translate, so it was removed.
 */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);

  const init = async () => {
    try {
      const health = await (await fetch("/api/health")).json();
      const kb = health.knowledge_base || {};
      $("#stat-diseases").textContent = kb.diseases ?? "—";
      $("#stat-symptoms").textContent = kb.symptoms ?? "—";
      $("#stat-medicines").textContent = kb.medicines ?? "—";
    } catch { /* leave the dashes */ }

    try {
      const data = await (await fetch("/api/languages")).json();
      const count = (data.languages || []).length;
      const el = $("#stat-languages");
      if (el && count) el.textContent = count;
    } catch { /* leave the dash */ }
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
