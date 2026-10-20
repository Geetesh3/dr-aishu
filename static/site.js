/* Dr. Aishu — shared site chrome: navigation, language switcher and theme.
 *
 * Loaded on every page. The language switcher is built to feel instantaneous:
 * every language's string table is fetched once (28 kB gzipped for all 23) and
 * held in memory, so selecting a language re-renders from memory with no
 * network round trip. The previous implementation issued a request per switch,
 * which is what made switching feel slow on a poor connection.
 *
 * Strings are applied in two ways:
 *   1. Static labels carrying data-i18n are swapped for the translated string.
 *   2. The choice is exposed as window.DrAishu, which page scripts read when
 *      they need to label something they build in JavaScript.
 *
 * A language is only offered once its safety strings exist (safety_complete),
 * because switching to a language that cannot render the emergency advice
 * would be worse than staying in English.
 */
(() => {
  "use strict";

  const STORE_KEY = "draishu_lang";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const getStored = () => {
    try { return localStorage.getItem(STORE_KEY) || "auto"; } catch { return "auto"; }
  };
  const setStored = (value) => {
    try { localStorage.setItem(STORE_KEY, value); } catch { /* private mode */ }
  };

  const state = {
    choice: getStored(),      // what the user picked: "auto" or a code
    code: "en",               // the code actually applied
    rtl: false,
    strings: {},              // resolved table for the applied code
    bundles: null,            // every language's table, fetched once
    catalogue: [],
    ready: false,
  };

  /* ---------- applying a language ---------- */

  const applyStrings = (root = document) => {
    $$("[data-i18n]", root).forEach((el) => {
      const value = state.strings[el.dataset.i18n];
      if (value) el.textContent = value;
    });
    $$("[data-i18n-placeholder]", root).forEach((el) => {
      const value = state.strings[el.dataset.i18nPlaceholder];
      if (value) el.setAttribute("placeholder", value);
    });
    $$("[data-i18n-title]", root).forEach((el) => {
      const value = state.strings[el.dataset.i18nTitle];
      if (value) el.setAttribute("title", value);
    });
    $$("[data-i18n-aria]", root).forEach((el) => {
      const value = state.strings[el.dataset.i18nAria];
      if (value) el.setAttribute("aria-label", value);
    });
  };

  const applyDirection = () => {
    const html = document.documentElement;
    html.setAttribute("dir", state.rtl ? "rtl" : "ltr");
    html.setAttribute("lang", state.code);
  };

  /** Resolve the table for a code, falling back to English. */
  const tableFor = (code) => {
    if (!state.bundles) return {};
    return state.bundles[code] || state.bundles.en || {};
  };

  /** Switch language. Reads from memory when the bundles are loaded. */
  const apply = async (choice, { silent = false } = {}) => {
    state.choice = choice;
    setStored(choice);

    let code = choice;
    if (choice === "auto") {
      code = document.documentElement.dataset.detected || "en";
    }

    // If bundles are still loading, fetch just this language so the first
    // paint is correct; otherwise resolve locally with no request.
    if (state.bundles && state.bundles[code]) {
      state.strings = tableFor(code);
      state.code = code;
      const meta = state.catalogue.find((l) => l.code === code);
      state.rtl = !!(meta && meta.rtl);
    } else {
      try {
        const res = await fetch(`/api/languages/${encodeURIComponent(code)}`);
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        state.strings = data.strings || {};
        state.rtl = !!data.rtl;
        state.code = data.code;
      } catch {
        state.strings = {};
        state.rtl = false;
        state.code = "en";
      }
    }

    applyStrings();
    applyDirection();
    updatePickerLabel();
    if (!silent) {
      document.dispatchEvent(new CustomEvent("draishu:lang", { detail: { ...state } }));
    }
  };

  /* ---------- the picker ---------- */

  const updatePickerLabel = () => {
    const select = $("#lang-select");
    if (select && select.value !== state.choice) select.value = state.choice;
  };

  const buildPicker = () => {
    const select = $("#lang-select");
    if (!select) return;

    const auto = select.querySelector('option[value="auto"]');
    select.innerHTML = "";
    if (auto) select.appendChild(auto);
    else {
      const opt = document.createElement("option");
      opt.value = "auto";
      opt.textContent = "Auto-detect";
      opt.dataset.i18n = "lang.auto";
      select.appendChild(opt);
    }

    const frag = document.createDocumentFragment();
    state.catalogue.forEach((lang) => {
      if (lang.code === "en") return;
      const opt = document.createElement("option");
      opt.value = lang.code;
      // Native name first: a Hindi speaker looks for "हिन्दी", not "Hindi".
      opt.textContent = `${lang.native} — ${lang.english}`;
      if (!lang.safety_complete) {
        opt.disabled = true;
        opt.textContent += " (unavailable)";
      }
      frag.appendChild(opt);
    });
    select.appendChild(frag);
    updatePickerLabel();
  };

  const loadCatalogue = async () => {
    const res = await fetch("/api/languages");
    const data = await res.json();
    state.catalogue = data.languages || [];
  };

  /** Warm the in-memory tables so the first switch is already instant. */
  const preloadBundles = async () => {
    try {
      const res = await fetch("/api/languages/all");
      if (!res.ok) return;
      const data = await res.json();
      state.bundles = data.bundles || null;
      // Re-apply now that switching no longer needs the network.
      if (state.bundles && state.choice !== "auto") await apply(state.choice, { silent: true });
    } catch { /* offline: per-switch fetch remains the fallback */ }
  };

  /* ---------- expose to page scripts ---------- */
  window.DrAishu = {
    get lang() { return state.choice === "auto" ? "auto" : state.code; },
    get code() { return state.code; },
    get rtl() { return state.rtl; },
    // A getter, not a snapshot: page scripts that capture this reference keep
    // seeing the current table after a language change.
    get strings() { return state.strings; },
    get catalogue() { return state.catalogue; },
    setLanguage: (choice) => apply(choice),
    t: (key, fallback = "") => state.strings[key] || fallback,
  };

  const init = async () => {
    const toggle = $("#nav-toggle");
    const menu = $("#header-menu");
    if (toggle && menu) {
      const setOpen = (open) => {
        menu.classList.toggle("is-open", open);
        toggle.setAttribute("aria-expanded", String(open));
      };
      toggle.addEventListener("click", () => setOpen(!menu.classList.contains("is-open")));
      // A tap on a link navigates; without this the menu stays open behind the
      // new page on a slow connection, which looks like nothing happened.
      menu.querySelectorAll(".header-nav a").forEach((a) =>
        a.addEventListener("click", () => setOpen(false)));
      document.addEventListener("click", (e) => {
        if (!menu.classList.contains("is-open")) return;
        if (!menu.contains(e.target) && !toggle.contains(e.target)) setOpen(false);
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && menu.classList.contains("is-open")) {
          setOpen(false);
          toggle.focus();
        }
      });
      // Leaving the mobile breakpoint must not strand an open dropdown over the
      // desktop layout, where the menu is always visible anyway.
      const wide = window.matchMedia("(min-width: 861px)");
      const onWide = (e) => { if (e.matches) setOpen(false); };
      if (wide.addEventListener) wide.addEventListener("change", onWide);
      else wide.addListener(onWide);
    }

    const select = $("#lang-select");
    if (select) select.addEventListener("change", (e) => apply(e.target.value));

    // Highlight the current page in the nav.
    const active = document.body.dataset.page;
    $$("[data-nav]").forEach((link) => {
      if (link.dataset.nav === active) link.classList.add("is-current");
    });

    // Apply the stored choice first so the page is readable immediately, then
    // build the picker and warm the tables.
    await apply(state.choice, { silent: true });
    try { await loadCatalogue(); } catch { /* keep the server-rendered list */ }
    buildPicker();
    state.ready = true;
    document.dispatchEvent(new CustomEvent("draishu:ready", { detail: { ...state } }));
    preloadBundles();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
