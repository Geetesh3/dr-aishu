/* Health AI — client logic. Vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const els = {
    form: $("#assess-form"),
    text: $("#symptom-text"),
    imageText: $("#image-text"),
    imageInput: $("#image-input"),
    dropzone: $("#dropzone"),
    preview: $("#preview"),
    previewImg: $("#preview-img"),
    previewName: $("#preview-name"),
    clearImage: $("#clear-image"),
    submit: $("#submit-btn"),
    reset: $("#reset-btn"),
    status: $("#status"),
    results: $("#results"),
    librarySearch: $("#library-search"),
    libraryList: $("#library-list"),
  };

  let activeTab = "text";
  let selectedFile = null;

  /* ---------- auth ----------
   * The hosted demo works with no key: the server issues a per-visitor demo key.
   * If the visitor has signed up at /developers, that key is picked up here and
   * unlocks the Pro-only actions (training, export).
   */
  const API_KEY = (() => {
    try { return sessionStorage.getItem("draishu_dev_key") || null; } catch { return null; }
  })();

  const apiFetch = (url, options = {}) => {
    if (!API_KEY) return fetch(url, options);
    const headers = Object.assign({}, options.headers || {});
    headers["X-API-Key"] = API_KEY;
    return fetch(url, Object.assign({}, options, { headers }));
  };

  /* ---------- helpers ---------- */
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const setStatus = (msg, kind = "loading") => {
    if (!msg) { els.status.hidden = true; els.status.textContent = ""; return; }
    els.status.hidden = false;
    els.status.className = `status is-${kind}`;
    els.status.textContent = msg;
  };

  /* ---------- tabs ---------- */
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      activeTab = tab.dataset.tab;
      $$(".tab").forEach((t) => {
        const on = t === tab;
        t.classList.toggle("is-active", on);
        t.setAttribute("aria-selected", String(on));
      });
      $$(".panel").forEach((p) => p.classList.toggle("is-active", p.dataset.panel === activeTab));
    });
  });

  /* ---------- example chips ---------- */
  $$(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.text.value = chip.dataset.example;
      els.text.focus();
    });
  });

  /* ---------- image selection ---------- */
  const showPreview = (file) => {
    selectedFile = file;
    els.previewName.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
    els.preview.hidden = false;
    els.dropzone.hidden = true;
    const reader = new FileReader();
    reader.onload = (e) => { els.previewImg.src = e.target.result; };
    reader.readAsDataURL(file);
  };

  const clearImage = () => {
    selectedFile = null;
    els.imageInput.value = "";
    els.preview.hidden = true;
    els.previewImg.removeAttribute("src");
    els.dropzone.hidden = false;
  };

  els.imageInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) showPreview(file);
  });
  els.clearImage.addEventListener("click", clearImage);

  ["dragenter", "dragover"].forEach((evt) =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.add("is-dragover");
    }));
  ["dragleave", "drop"].forEach((evt) =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("is-dragover");
    }));
  els.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) showPreview(file);
  });

  /* ---------- submit ---------- */
  els.form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = activeTab === "text" ? els.text.value.trim() : els.imageText.value.trim();
    if (!text && !selectedFile) {
      setStatus("Describe your symptoms, upload an image, or do both.", "error");
      return;
    }

    els.submit.disabled = true;
    setStatus("Analysing…", "loading");
    els.results.hidden = true;

    try {
      let res;
      if (selectedFile) {
        const fd = new FormData();
        fd.append("image", selectedFile);
        fd.append("text", text);
        fd.append("top_n", "6");
        fd.append("include_rx", String($("#ctx-rx").checked));
        fd.append("consent_share", "true");
        const age = $("#ctx-age").value;
        const weight = $("#ctx-weight").value;
        if (age) fd.append("age", age);
        if (weight) fd.append("weight_kg", weight);
        if ($("#ctx-pregnant").checked) fd.append("pregnant", "true");
        fd.append("chronic_conditions", $("#ctx-conditions").value);
        fd.append("allergies", $("#ctx-allergies").value);
        fd.append("current_medicines", ($("#ctx-current-meds") || {}).value || "");
        fd.append("lang", (window.DrAishu && window.DrAishu.lang) || "auto");
        var modality = ($("#image-modality") || {}).value;
        if (modality) fd.append("image_modality", modality);
        res = await apiFetch("/api/analyze-image", { method: "POST", body: fd });
      } else {
        res = await apiFetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            top_n: 6,
            include_rx: $("#ctx-rx").checked,
            consent_share: true,
            age: $("#ctx-age").value ? Number($("#ctx-age").value) : null,
            weight_kg: $("#ctx-weight").value ? Number($("#ctx-weight").value) : null,
            pregnant: $("#ctx-pregnant").checked,
            chronic_conditions: $("#ctx-conditions").value,
            allergies: $("#ctx-allergies").value,
            current_medicines: ($("#ctx-current-meds") || {}).value || "",
            lang: (window.DrAishu && window.DrAishu.lang) || "auto",
          }),
        });
      }

      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try { detail = (await res.json()).detail || detail; } catch (_) {}
        throw new Error(detail);
      }

      render(await res.json());
      setStatus("");
      els.results.hidden = false;
      els.results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setStatus(err.message || "Something went wrong. Please try again.", "error");
    } finally {
      els.submit.disabled = false;
    }
  });

  els.reset.addEventListener("click", () => {
    setTimeout(() => {
      clearImage();
      els.results.hidden = true;
      setStatus("");
    }, 0);
  });

  /* ---------- rendering ---------- */
  function render(data) {
    const sev = data.severity || {};
    const dx = data.diagnoses || [];
    const meds = data.medicines || {};
    const routine = data.routine || {};

    const parts = [];

    parts.push(sectionTriage(sev, dx));
    if (data.image_analysis) parts.push(sectionImage(data.image_analysis));
    parts.push(sectionExtracted(data.extracted));
    parts.push(sectionSolutions(data.solutions));
    parts.push(sectionDiagnoses(dx, data));
    parts.push(sectionMedicines(meds, sev));
    parts.push(sectionRoutine(routine));
    parts.push(`<div class="notice">${esc(data.disclaimer || "")}</div>`);

    els.results.innerHTML = parts.join("");
    wireDiagnosesSlider();
    wireFeedback(data);
  }

  /* ---------- feedback: teaches Dr. Aishu the correct answer ---------- */
  function wireFeedback(data) {
    const caseId = data.case && data.case.case_id;
    const box = $("#feedback-box");
    if (!box || !caseId) return;

    const options = (data.diagnoses || [])
      .map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
    const allDx = box.dataset.allDiagnoses || "";

    $("#fb-submit").addEventListener("click", async () => {
      const correct = $("#fb-correct").value;
      const rating = $("#fb-rating").value;
      const comment = $("#fb-comment").value;
      const note = $("#fb-note");

      if (!correct && !rating && !comment.trim()) {
        note.textContent = "Choose a correct diagnosis, a rating, or add a comment first.";
        note.hidden = false;
        return;
      }

      try {
        const res = await apiFetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            case_id: caseId,
            correct_diagnosis: correct || null,
            rating: rating ? Number(rating) : null,
            comment: comment.trim() || null,
          }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || "Could not save feedback");
        note.textContent = correct
          ? "Thank you — Dr. Aishu will learn from this correction at the next training run."
          : "Thank you — your feedback was recorded.";
        note.hidden = false;
        $("#fb-submit").disabled = true;
      } catch (err) {
        note.textContent = err.message;
        note.hidden = false;
      }
    });

    const stored = data.case && data.case.stored_for_training;
    $("#fb-store-state").textContent = stored
      ? "This case was stored anonymously and will be used for training."
      : "This case could not be stored for training.";
    $("#fb-store-state").className = stored ? "notice" : "notice warn";
  }

  function sectionTriage(sev, dx) {
    const flags = (sev.red_flags || []).map((f) => `<li>${esc(f)}</li>`).join("");
    const factors = (sev.factors || [])
      .map((f) => `<li><strong>${esc(f.factor)}</strong> — ${esc(f.impact)}</li>`)
      .join("");
    const da = sev.duration_assessment || {};
    const durationLine = da.reported_days
      ? `<div class="notice">Reported duration <strong>${da.reported_days} day(s)</strong> · expected for ${esc(dx[0] ? dx[0].name : "this condition")}: ${da.expected_days ? da.expected_days.map((n) => Math.round(n)).join("–") : "—"} days · <strong>${esc(da.status || "")}</strong></div>`
      : "";

    return `
      <section class="card">
        <div class="result-head">
          <h2 style="margin:0">Severity &amp; triage</h2>
        </div>
        <div class="triage" data-colour="${esc(sev.colour || "green")}">
          <div class="triage-score">${esc(sev.score ?? "—")}<span style="font-size:.9rem;font-weight:400">/10</span></div>
          <div class="triage-body">
            <span class="triage-label">${esc(sev.triage_label || "")}</span>
            <span class="triage-advice">${esc(sev.advice || "")}</span>
          </div>
        </div>
        ${flags ? `<h3>Warning signs detected</h3><ul class="red-flags">${flags}</ul>` : ""}
        ${durationLine}
        ${factors ? `<h3>How this score was reached</h3><ul class="plain">${factors}</ul>` : ""}
      </section>`;
  }

  function sectionImage(img) {
    if (!img || !img.ok) {
      return `<section class="card"><h2>Image analysis</h2>
        <div class="notice warn">${esc((img && img.error) || "Image could not be processed.")}</div></section>`;
    }
    const obs = (img.observations || [])
      .map((o) => `<li><strong>${esc(o.observation)}</strong>${o.detail ? ` — ${esc(o.detail)}` : ""} <span class="muted">(confidence ${(o.confidence * 100).toFixed(0)}%)</span></li>`)
      .join("");
    const ocr = img.ocr || {};
    const ocrBlock = ocr.text
      ? `<h3>Text read from the image</h3>
         <div class="notice">${esc(ocr.text.slice(0, 1200))}${ocr.text.length > 1200 ? "…" : ""}</div>
         <p class="muted">OCR confidence: ${ocr.confidence != null ? esc(ocr.confidence) + "%" : "not available"}</p>`
      : `<p class="muted">No readable text was found in this image.</p>`;
    const vf = img.visual_features || {};
    const mp = img.model_prediction;
    const modelBlock = renderImageModel(mp);

    return `
      <section class="card">
        <h2>Image analysis</h2>
        <p class="muted">${esc(img.image.width)}×${esc(img.image.height)} px · dominant tone: ${esc(vf.dominant_hue || "—")} · document-like: ${vf.is_likely_document ? "yes" : "no"}</p>
        ${modelBlock}
        <h3>Visual observations</h3>
        <ul class="plain">${obs || "<li>No notable features.</li>"}</ul>
        ${ocrBlock}
        <p class="muted">Image analysis describes visible features only. It cannot confirm a diagnosis on its own.</p>
      </section>`;
  }

  function renderImageModel(mp) {
    if (!mp) return "";
    if (mp.ok === false) {
      return `<div class="notice">${esc(mp.error || "No trained model matched this image.")}</div>`;
    }
    // Without a modality the model returns every candidate and refuses to pick,
    // because an eye model is overconfident on a skin photo. Showing that raw
    // output would put a confident wrong answer in front of the user, so the
    // candidates are only listed once the modality is known.
    const preds = mp.modality && mp.predictions
      ? (mp.predictions[mp.modality].top || []).slice(0, 3)
          .map((t) => `<li>${esc(t.label)} <span class="muted">(${(t.confidence * 100).toFixed(0)}%)</span></li>`)
          .join("")
      : "";
    const headline = mp.uncertain
      ? `<p class="notice warn">The model could not match this image to a condition it is confident about. Tell us what the photo shows (skin or eye) to get a sharper result.</p>`
      : `<p><strong>Closest visual match:</strong> ${esc(mp.label)} <span class="muted">(${(mp.confidence * 100).toFixed(0)}% confidence${mp.modality ? ", " + esc(mp.modality) : ""})</span></p>`;
    const list = preds ? `<ul class="plain">${preds}</ul>` : "";
    return `<h3>Trained model</h3>${headline}${list}
      <p class="muted">A visual match is not a diagnosis. It only says which of the conditions the model has seen looks most like this photo.</p>`;
  }

  function sectionExtracted(ex) {
    if (!ex) return "";
    const chips = (arr, cls = "tag") =>
      (arr || []).map((x) => `<span class="${cls}">${esc(x.label || x)}</span>`).join("");
    const dem = ex.demographics || {};
    const vit = ex.vitals || {};
    const hist = ex.history || {};
    const dur = ex.duration || {};
    const sevSelf = ex.self_reported_severity || {};

    const vitals = Object.entries({
      "Temperature": vit.temperature_c != null ? `${vit.temperature_c} °C` : null,
      "Oxygen saturation": vit.oxygen_saturation != null ? `${vit.oxygen_saturation}%` : null,
      "Blood pressure": vit.systolic_bp != null ? `${vit.systolic_bp}/${vit.diastolic_bp} mmHg` : null,
      "Pulse": vit.pulse != null ? `${vit.pulse} bpm` : null,
      "Duration": dur.days != null ? `${dur.days} day(s)` : null,
      "Self-rated severity": sevSelf.score != null ? `${sevSelf.score}/10` : null,
      "Age": dem.age != null ? `${dem.age} years` : null,
      "Weight": dem.weight_kg != null ? `${dem.weight_kg} kg` : null,
    })
      .filter(([, v]) => v != null)
      .map(([k, v]) => `<span class="tag">${esc(k)}: <strong>${esc(v)}</strong></span>`)
      .join("");

    return `
      <section class="card">
        <h2>What was understood</h2>
        <h3>Symptoms detected</h3>
        <div class="dx-meta">${chips(ex.symptoms, "tag matched") || '<span class="muted">None</span>'}</div>
        ${(ex.negated_symptoms || []).length ? `<h3>Explicitly denied</h3><div class="dx-meta">${chips(ex.negated_symptoms)}</div>` : ""}
        ${(ex.modifiers || []).length ? `<h3>Onset / pattern</h3><div class="dx-meta">${chips(ex.modifiers)}</div>` : ""}
        ${vitals ? `<h3>Measurements and context</h3><div class="dx-meta">${vitals}</div>` : ""}
        ${(hist.chronic_conditions || []).length ? `<h3>Chronic conditions</h3><div class="dx-meta">${chips(hist.chronic_conditions)}</div>` : ""}
        ${(hist.allergies || []).length ? `<h3>Allergies</h3><div class="dx-meta">${chips(hist.allergies)}</div>` : ""}
      </section>`;
  }

  /* ---------- per-problem action plan for the leading condition ---------- */
  function sectionSolutions(sol) {
    if (!sol || (!sol.problems || !sol.problems.length) && !sol.condition) return "";
    const problems = sol.problems || [];

    const cards = problems.map((p) => {
      const steps = (p.steps || []).map((s) => `<li>${esc(s)}</li>`).join("");
      const covered = (p.covered_by || []).length
        ? `<div class="sol-covered">Covered by your plan: ${(p.covered_by || []).map(esc).join(", ")}</div>`
        : "";
      const flag = p.is_red_flag
        ? `<span class="tag flag">${esc(p.red_flag_level || "warning")} sign</span>`
        : p.is_key ? '<span class="tag matched">key symptom</span>' : "";
      const note = p.note ? `<p class="sol-note">${esc(p.note)}</p>` : "";
      return `
        <div class="sol-problem">
          <div class="sol-head">
            <strong>${esc(p.label)}</strong>
            ${flag}
          </div>
          ${steps ? `<ul class="plain">${steps}</ul>` : ""}
          ${covered}
          ${note}
        </div>`;
    }).join("");

    const notes = (sol.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");
    const cond = sol.condition;

    return `
      <section class="card">
        <h2>What to do about each problem</h2>
        ${cond ? `<p class="muted">Supportive steps for <strong>${esc(cond.name)}</strong>, ordered with any warning signs first.</p>`
               : `<p class="muted">Supportive steps for what you reported.</p>`}
        ${problems.length ? `<div class="sol-grid">${cards}</div>`
                          : `<div class="notice warn">No specific steps were recorded for the reported symptoms.</div>`}
        ${notes ? `<h3>Worth knowing</h3><ul class="plain">${notes}</ul>` : ""}
      </section>`;
  }



  function sectionDiagnoses(dx, data) {
    if (!dx.length) {
      return `<section class="card"><h2>Likely conditions</h2>
        <div class="notice warn">No condition matched with confidence. Add more detail about your symptoms.</div>
        ${feedbackBox([], data)}</section>`;
    }
    // One slide per condition, so the whole differential occupies the height of
    // a single card. Detail that is not needed at a glance stays behind the
    // Read more control.
    const slides = dx.map((d, i) => {
      const matched = (d.matched_symptoms || []).map((s) => `<span class="tag matched">${esc(s)}</span>`).join("");
      const missing = (d.missing_key_symptoms || []).map((s) => `<span class="tag">${esc(s)}</span>`).join("");
      const care = (d.self_care || []).map((s) => `<li>${esc(s)}</li>`).join("");
      const doctor = (d.when_to_see_doctor || []).map((s) => `<li>${esc(s)}</li>`).join("");
      const dur = d.typical_duration_days
        ? `${Math.round(d.typical_duration_days[0])}–${Math.round(d.typical_duration_days[1])} days`
        : "—";
      const declared = d.declared_condition
        ? `<span class="tag declared" title="You told us you already have this">your condition</span>`
        : "";
      const hasDetail = matched || missing || care || doctor;
      return `
        <article class="dx-slide" data-index="${i}">
          <div class="dx-top">
            <span class="dx-rank">${i + 1}</span>
            <span class="dx-name">${esc(d.name)}</span>
            <span class="dx-conf">${esc(d.confidence)}%</span>
          </div>
          <div class="dx-bar"><span style="width:${Math.max(3, Math.min(100, d.confidence))}%"></span></div>
          <div class="dx-meta">
            ${declared}
            <span class="tag">${esc(d.category)}</span>
            ${d.icd10 ? `<span class="tag">ICD-10 ${esc(d.icd10)}</span>` : ""}
            <span class="tag">typical ${esc(dur)}</span>
            ${d.contagious ? '<span class="tag">contagious</span>' : ""}
          </div>
          <p class="dx-desc">${esc(d.description)}</p>
          ${hasDetail ? `
            <div class="dx-detail" hidden>
              ${matched ? `<div class="dx-sub">Matching symptoms</div><div class="dx-meta">${matched}</div>` : ""}
              ${missing ? `<div class="dx-sub">Key symptoms not reported</div><div class="dx-meta">${missing}</div>` : ""}
              ${care ? `<div class="dx-sub">Self-care</div><ul>${care}</ul>` : ""}
              ${doctor ? `<div class="dx-sub">See a doctor if</div><ul>${doctor}</ul>` : ""}
            </div>
            <button type="button" class="dx-more" aria-expanded="false">Read more</button>
          ` : ""}
        </article>`;
    }).join("");

    const dots = dx.map((_, i) =>
      `<button type="button" class="dx-dot${i === 0 ? " active" : ""}" data-goto="${i}" aria-label="Condition ${i + 1}"></button>`
    ).join("");

    return `<section class="card">
      <div class="dx-head">
        <h2>Likely conditions (ranked)</h2>
        <div class="dx-nav">
          <button type="button" class="dx-arrow" id="dx-prev" aria-label="Previous condition">‹</button>
          <span class="dx-counter" id="dx-counter">1 / ${dx.length}</span>
          <button type="button" class="dx-arrow" id="dx-next" aria-label="Next condition">›</button>
        </div>
      </div>
      <p class="muted dx-hint">Confidence is relative between the listed conditions, not a probability of having the disease.</p>
      <div class="dx-slider" id="dx-slider">
        <div class="dx-track" id="dx-track">${slides}</div>
      </div>
      <div class="dx-dots" id="dx-dots">${dots}</div>
      ${feedbackBox(dx, data)}</section>`;
  }

  function wireDiagnosesSlider() {
    const track = $("#dx-track");
    if (!track) return;
    const slides = Array.from(track.querySelectorAll(".dx-slide"));
    const dots = Array.from(document.querySelectorAll("#dx-dots .dx-dot"));
    const counter = $("#dx-counter");
    let index = 0;

    const show = (i) => {
      index = Math.max(0, Math.min(slides.length - 1, i));
      track.style.transform = `translateX(${-index * 100}%)`;
      dots.forEach((d, n) => d.classList.toggle("active", n === index));
      if (counter) counter.textContent = `${index + 1} / ${slides.length}`;
      const prev = $("#dx-prev");
      const next = $("#dx-next");
      if (prev) prev.disabled = index === 0;
      if (next) next.disabled = index === slides.length - 1;
    };

    const prev = $("#dx-prev");
    const next = $("#dx-next");
    if (prev) prev.addEventListener("click", () => show(index - 1));
    if (next) next.addEventListener("click", () => show(index + 1));
    dots.forEach((d, n) => d.addEventListener("click", () => show(n)));

    // "Read more" expands only its own slide, so the compact height is kept for
    // every condition the reader has not opened.
    slides.forEach((slide) => {
      const btn = slide.querySelector(".dx-more");
      const detail = slide.querySelector(".dx-detail");
      if (!btn || !detail) return;
      btn.addEventListener("click", () => {
        const open = detail.hidden;
        detail.hidden = !open;
        btn.textContent = open ? "Read less" : "Read more";
        btn.setAttribute("aria-expanded", String(open));
      });
    });

    // Trackpad and touch swipes move between conditions.
    let startX = null;
    track.addEventListener("touchstart", (e) => { startX = e.touches[0].clientX; }, { passive: true });
    track.addEventListener("touchend", (e) => {
      if (startX === null) return;
      const delta = e.changedTouches[0].clientX - startX;
      if (Math.abs(delta) > 40) show(index + (delta < 0 ? 1 : -1));
      startX = null;
    });

    show(0);
  }

  function feedbackBox(dx, data) {
    const caseId = data && data.case && data.case.case_id;
    if (!caseId) return "";
    const opts = ["<option value=\"\">Select the correct condition…</option>"]
      .concat(dx.map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`))
      .join("");
    const modelNote = data.model_predictions && data.model_predictions.length
      ? `<p class="muted">Dr. Aishu's trained model also considered: ${data.model_predictions.slice(0, 3).map((p) => `${esc(p.label)} (${(p.probability * 100).toFixed(0)}%)`).join(", ")}.</p>`
      : "";
    return `
      <div class="feedback" id="feedback-box">
        <h3>Was this helpful?</h3>
        ${modelNote}
        <div class="feedback-grid">
          <label class="field"><span class="field-label">Correct condition, if different</span>
            <select id="fb-correct">${opts}</select></label>
          <label class="field"><span class="field-label">Rating</span>
            <select id="fb-rating">
              <option value="">No rating</option>
              <option value="5">5 — very helpful</option>
              <option value="4">4 — helpful</option>
              <option value="3">3 — neutral</option>
              <option value="2">2 — not helpful</option>
              <option value="1">1 — poor</option>
            </select></label>
        </div>
        <label class="field"><span class="field-label">Comments <em>(optional)</em></span>
          <textarea id="fb-comment" rows="2" placeholder="Anything Dr. Aishu should know about this case?"></textarea></label>
        <div class="actions">
          <button type="button" class="btn-ghost" id="fb-submit">Send feedback</button>
        </div>
        <p class="muted" id="fb-note" hidden></p>
        <p class="notice" id="fb-store-state"></p>
      </div>`;
  }

  function sectionMedicines(meds, sev) {
    if (sev.triage === "emergency") {
      return `<section class="card"><h2>Medicines</h2>
        <div class="notice warn"><strong>Do not take any new medicine now.</strong> Call emergency services or go to the nearest emergency department.</div>
        <ul class="plain">${(meds.general_advice || []).map((a) => `<li>${esc(a)}</li>`).join("")}</ul>
      </section>`;
    }

    const rec = (meds.recommended || []).map((m) => medCard(m, "otc")).join("");
    const rx = (meds.prescription_only || []).map((m) => medCard(m, "rx")).join("");
    const inter = (meds.interactions || []).map((i) =>
      `<li><strong>${esc(i.between.join(" + "))}</strong> — ${esc(i.note)}</li>`).join("");
    const excluded = (meds.excluded || []).map((e) =>
      `<li><strong>${esc(e.name)}</strong> — ${esc(e.reason)}</li>`).join("");
    const advice = (meds.general_advice || []).map((a) => `<li>${esc(a)}</li>`).join("");

    const tt = (meds.timetable || []).map((slot) => `
      <div class="slot">
        <div class="slot-head">${esc(slot.label)}</div>
        <div class="slot-rows">
          ${slot.entries.map((e) => `
            <div class="slot-row">
              <span class="slot-time">${esc(e.time || "PRN")}</span>
              <span>
                <span class="slot-med">${esc(e.medicine)}</span>
                <span class="slot-dose"> · ${esc(e.dose || e.instruction || "")}${e.with_food && e.with_food !== "any" ? ` · ${esc(e.with_food)}` : ""}</span>
              </span>
              <span class="slot-badge ${e.otc ? "otc" : "rx"}">${e.otc ? "OTC" : "Rx"}</span>
            </div>`).join("")}
        </div>
      </div>`).join("");

    return `
      <section class="card">
        <h2>Medicine suggestions</h2>
        <div class="notice">These are general suggestions based on the likely conditions, filtered against the allergies and conditions you entered. They are not a prescription. Confirm every medicine with a pharmacist or doctor.</div>

        ${rec ? `<h3>Over-the-counter options</h3><div class="grid-2">${rec}</div>` : `<p class="muted">No over-the-counter options were suitable.</p>`}
        ${rx ? `<h3>Prescription-only options to discuss with a clinician</h3><div class="notice rx">These require a clinician's authorisation. Do not take them without a prescription.</div><div class="grid-2">${rx}</div>` : ""}

        <h3>Medicine timetable</h3>
        ${tt || '<p class="muted">No scheduled medicines.</p>'}

        ${inter ? `<h3>Potential interactions</h3><ul class="plain">${inter}</ul>` : ""}
        ${excluded ? `<h3>Filtered out for safety</h3><ul class="plain">${excluded}</ul>` : ""}
        ${advice ? `<h3>Safety advice</h3><ul class="plain">${advice}</ul>` : ""}
      </section>`;
  }

  function medCard(m, kind) {
    return `
      <article class="dx">
        <div class="dx-top">
          <span class="dx-name">${esc(m.name)}</span>
          <span class="slot-badge ${kind}">${kind === "otc" ? "OTC" : "Rx"}</span>
        </div>
        <div class="dx-meta">
          <span class="tag">${esc(m.drug_class)}</span>
          <span class="tag">${esc(m.dose)}</span>
          <span class="tag">${esc(m.frequency)}</span>
          ${m.with_food && m.with_food !== "any" ? `<span class="tag">${esc(m.with_food)}</span>` : ""}
          ${m.max_days ? `<span class="tag">max ${esc(m.max_days)} days</span>` : ""}
        </div>
        <p class="dx-desc">${esc(m.reason)}${m.for_conditions && m.for_conditions.length ? ` · for ${esc(m.for_conditions.join(", "))}` : ""}</p>
        ${m.warnings && m.warnings.length ? `<div class="dx-sub">Cautions</div><ul>${m.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
        ${m.side_effects && m.side_effects.length ? `<div class="dx-sub">Possible side effects</div><ul>${m.side_effects.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
      </article>`;
  }

  function sectionRoutine(r) {
    if (!r || !r.blocks) return "";
    const blocks = r.blocks.map((b) => `
      <div class="day-block" data-category="${esc(b.category || "")}">
        <span class="day-time">${esc(b.time)}</span>
        <span>
          <span class="day-title">${esc(b.title)}</span>
          <span class="day-detail">${esc(b.detail)}</span>
        </span>
      </div>`).join("");
    const habits = (r.habits || []).map((h) => `<li>${esc(h)}</li>`).join("");
    const avoid = (r.avoid || []).map((h) => `<li>${esc(h)}</li>`).join("");
    const notes = (r.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");

    return `
      <section class="card">
        <h2>Your daily routine</h2>
        <div class="routine-head">
          <div>
            <strong>${esc(r.name)}</strong>
            <p class="muted" style="margin:.15rem 0 0">${esc(r.focus)}</p>
          </div>
          <div class="routine-targets">
            <span><strong>${esc(r.hydration_target_l)} L</strong><br>water</span>
            <span><strong>${esc(r.sleep_target_hours)} h</strong><br>sleep</span>
          </div>
        </div>
        <div class="day">${blocks}</div>
        ${notes ? `<h3>Personalised notes</h3><ul class="plain">${notes}</ul>` : ""}
        <div class="two-col">
          ${habits ? `<div><h3>Daily habits</h3><ul class="plain">${habits}</ul></div>` : ""}
          ${avoid ? `<div><h3>Things to avoid</h3><ul class="plain">${avoid}</ul></div>` : ""}
        </div>
      </section>`;
  }

  /* ---------- knowledge base stats + library ---------- */
  let libraryCache = [];

  async function loadLibrary() {
    try {
      const [hRes, dRes] = await Promise.all([apiFetch("/api/health"), apiFetch("/api/diseases")]);
      const health = await hRes.json();
      const kb = health.knowledge_base || {};
      $("#stat-diseases").textContent = kb.diseases ?? "—";
      $("#stat-symptoms").textContent = kb.symptoms ?? "—";
      $("#stat-medicines").textContent = kb.medicines ?? "—";

      libraryCache = (await dRes.json()).diseases || [];
      renderLibrary(libraryCache);
    } catch (_) {
      els.libraryList.innerHTML = '<p class="muted">Could not load the condition library.</p>';
    }
  }

  function renderLibrary(list) {
    if (!list.length) {
      els.libraryList.innerHTML = '<p class="muted">No conditions match that search.</p>';
      return;
    }
    els.libraryList.innerHTML = list.map((d) => `
      <article class="lib-item">
        <span class="lib-cat">${esc(d.category)}${d.contagious ? " · contagious" : ""}</span>
        <h4>${esc(d.name)}</h4>
        <p>${esc(d.description)}</p>
        <p><strong>Symptoms:</strong> ${esc((d.symptoms || []).slice(0, 6).join(", "))}${(d.symptoms || []).length > 6 ? "…" : ""}</p>
      </article>`).join("");
  }

  // The library markup only exists on /library, but app.js is loaded on every
  // page that shares the assessment shell, so this must not assume the nodes.
  if (els.librarySearch && els.libraryList) {
    els.librarySearch.addEventListener("input", (e) => {
      const q = e.target.value.trim().toLowerCase();
      if (!q) return renderLibrary(libraryCache);
      renderLibrary(libraryCache.filter((d) =>
        d.name.toLowerCase().includes(q) ||
        d.category.toLowerCase().includes(q) ||
        d.description.toLowerCase().includes(q) ||
        (d.symptoms || []).join(" ").toLowerCase().includes(q)));
    });

    loadLibrary();
  }
})();
