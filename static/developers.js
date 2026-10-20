/* Developer portal: issue, inspect and revoke Dr. Aishu API keys.
 *
 * The plaintext key is held in sessionStorage only so the usage panel survives a
 * reload within the same tab. It is never written to localStorage and is dropped
 * when the tab closes.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const STORE_KEY = "draishu_dev_key";

  const status = (message, kind) => {
    const el = $("dev-status");
    if (!message) { el.hidden = true; return; }
    el.hidden = false;
    el.textContent = message;
    el.className = `status is-${kind || "loading"}`;
  };

  const api = async (path, options = {}) => {
    const key = sessionStorage.getItem(STORE_KEY);
    const headers = Object.assign({}, options.headers || {});
    if (key) headers["X-API-Key"] = key;
    const res = await fetch(path, Object.assign({}, options, { headers }));
    const text = await res.text();
    let body;
    try { body = text ? JSON.parse(text) : {}; } catch { body = { detail: text }; }
    if (!res.ok) {
      const detail = Array.isArray(body.detail)
        ? body.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
        : (body.detail || `Request failed (${res.status})`);
      const err = new Error(detail);
      err.status = res.status;
      throw err;
    }
    return body;
  };

  const method = (verb) => `<span class="method ${verb === "GET" ? "get" : "post"}">${verb}</span>`;

  // ---- tiers ---------------------------------------------------------------
  async function loadTiers() {
    try {
      const { tiers } = await api("/api/keys/tiers");
      $("tier-row").innerHTML = tiers.map((t) => `
        <div class="tier-card${t.id === "free" ? " is-featured" : ""}">
          <div class="tier-name">${t.label} <span class="tier-price">Free</span></div>
          <p class="tier-desc">${t.description}</p>
          <ul class="tier-limits">
            <li><span>Requests / minute</span><b>${t.per_minute}</b></li>
            <li><span>Requests / day</span><b>${t.per_day}</b></li>
            <li><span>Scopes</span><b>${t.scopes.length}</b></li>
          </ul>
        </div>`).join("");
    } catch (err) {
      status(`Could not load tiers: ${err.message}`, "error");
    }
  }

  // ---- issue a key ---------------------------------------------------------
  async function signup() {
    const btn = $("signup-btn");
    btn.disabled = true;
    status("Creating your key…", "loading");
    try {
      const body = await api("/api/keys/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: $("signup-email").value.trim() || null,
          label: $("signup-label").value.trim() || null,
          tier: "free",
        }),
      });
      sessionStorage.setItem(STORE_KEY, body.secret);
      $("key-secret").textContent = body.secret;
      $("key-meta").textContent =
        `Tier: ${body.tier_label} · ${body.limits.per_minute}/min · ${body.limits.per_day}/day · ` +
        `scopes: ${body.scopes.join(", ")}`;
      $("key-reveal").hidden = false;
      status("Key created. Copy it now — it will not be shown again.", "loading");
      if (body.email) $("lookup-email").value = body.email;
      await refreshUsage();
    } catch (err) {
      status(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  async function refreshUsage() {
    if (!sessionStorage.getItem(STORE_KEY)) return;
    try {
      const me = await api("/api/keys/me");
      $("usage-panel").hidden = false;
      $("u-tier").textContent = me.tier_label;
      $("u-minute").textContent = `${me.usage.minute.used} / ${me.usage.minute.limit}`;
      $("u-day").textContent = `${me.usage.day.used} / ${me.usage.day.limit}`;
      $("u-total").textContent = me.request_count;
      $("key-reveal").hidden = false;
      $("key-secret").textContent = sessionStorage.getItem(STORE_KEY);
      $("key-meta").textContent = `Tier: ${me.tier_label} · scopes: ${me.scopes.join(", ")}`;
    } catch (err) {
      if (err.status === 401) {
        sessionStorage.removeItem(STORE_KEY);
        $("usage-panel").hidden = true;
        $("key-reveal").hidden = true;
      } else {
        status(err.message, "error");
      }
    }
  }

  async function revoke() {
    if (!confirm("Revoke this key? Any app using it will start receiving 401.")) return;
    try {
      const me = await api("/api/keys/me");
      await api("/api/keys/revoke", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key_id: me.id }),
      });
      sessionStorage.removeItem(STORE_KEY);
      $("usage-panel").hidden = true;
      $("key-reveal").hidden = true;
      status("Key revoked.", "loading");
    } catch (err) {
      status(err.message, "error");
    }
  }

  async function lookup() {
    const email = $("lookup-email").value.trim();
    if (!email) { status("Enter the email you signed up with.", "error"); return; }
    try {
      const { keys } = await api(`/api/keys/list?email=${encodeURIComponent(email)}`);
      $("lookup-result").innerHTML = keys.length
        ? keys.map((k) => `
          <div class="key-row${k.revoked ? " is-revoked" : ""}">
            <div>
              <code>${k.key}</code>
              <div class="meta">${k.tier_label} · created ${k.created_at.slice(0, 10)} · ${k.request_count} calls${k.label ? ` · ${k.label}` : ""}</div>
            </div>
            <span class="pill ${k.revoked ? "pill-revoked" : "pill-ok"}">${k.revoked ? "revoked" : "active"}</span>
          </div>`).join("")
        : `<p class="muted">No keys found for that email.</p>`;
    } catch (err) {
      status(err.message, "error");
    }
  }

  // ---- copy helpers --------------------------------------------------------
  function wireCopy() {
    document.querySelectorAll(".copy-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const target = $(btn.dataset.copy);
        if (!target) return;
        try {
          await navigator.clipboard.writeText(target.innerText);
          btn.textContent = "Copied";
          setTimeout(() => { btn.textContent = "Copy"; }, 1400);
        } catch {
          status("Clipboard blocked by the browser; select and copy manually.", "error");
        }
      });
    });

    $("copy-key").addEventListener("click", async () => {
      const key = sessionStorage.getItem(STORE_KEY);
      if (!key) return;
      try {
        await navigator.clipboard.writeText(key);
        $("copy-key").textContent = "Copied";
        setTimeout(() => { $("copy-key").textContent = "Copy"; }, 1400);
      } catch {
        status("Clipboard blocked; select the key and copy manually.", "error");
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    wireCopy();
    loadTiers();
    refreshUsage();
    $("signup-btn").addEventListener("click", signup);
    $("refresh-usage").addEventListener("click", refreshUsage);
    $("revoke-btn").addEventListener("click", revoke);
    $("lookup-btn").addEventListener("click", lookup);
  });
})();
