const recForm = document.getElementById("rec-form");
const results = document.getElementById("results");
const recErr = document.getElementById("rec-error");
const recBtn = document.getElementById("rec-btn");

function listing(p) {
  const baths = Number.isInteger(p.bathrooms) ? p.bathrooms : p.bathrooms.toFixed(1);
  return `
  <article class="listing" data-id="${p.id}">
    <div>
      <div class="price">${money(p.price)}</div>
      <div class="where">${esc(p.neighborhood_name)}</div>
      <div class="specs">${p.bedrooms} bed &middot; ${baths} bath &middot; ${p.area.toLocaleString()} sq ft &middot; built ${p.year_built}${p.garage ? ` &middot; ${p.garage}-car garage` : ""}</div>
      <div class="match"><span class="bar"><i style="width:${Math.round(p.score)}%"></i></span><span>Match score ${Math.round(p.score)} of 100</span></div>
      <button class="btn ghost small" type="button" data-summary="${p.id}">Summarize with AI</button>
      <div class="ai-summary" hidden></div>
    </div>
    <div>
      <div class="why-title">Why it fits</div>
      <ul class="why">${p.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
    </div>
  </article>`;
}

recForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(recForm);
  const body = {
    budget: Number(f.get("budget")),
    neighborhood: f.get("neighborhood") || null,
    min_bedrooms: Number(f.get("min_bedrooms") || 0),
    min_bathrooms: Number(f.get("min_bathrooms") || 0),
    min_quality: Number(f.get("min_quality") || 1),
    min_area: Number(f.get("min_area") || 0),
    garage: document.getElementById("garage").checked,
    fireplace: document.getElementById("fireplace").checked,
  };
  recBtn.disabled = true; recBtn.textContent = "Searching...";
  recErr.hidden = true;
  try {
    const data = await postJSON("/api/recommend", body);
    if (!data.results.length) {
      results.innerHTML = `<p class="empty"><b>No homes match.</b> ${esc(data.hint || "Try loosening a filter.")}</p>`;
    } else {
      results.innerHTML =
        `<div class="result-head"><h2>Top ${data.results.length} of ${data.total_matches.toLocaleString()} matching homes</h2></div>` +
        data.results.map(listing).join("");
    }
  } catch (err) {
    recErr.textContent = err.message; recErr.hidden = false;
  } finally {
    recBtn.disabled = false; recBtn.textContent = "Find homes";
  }
});

results.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-summary]");
  if (!btn) return;
  const box = btn.parentElement.querySelector(".ai-summary");
  if (!box.hidden) { box.hidden = true; btn.textContent = "Summarize with AI"; return; }
  btn.disabled = true; btn.textContent = "Writing summary...";
  try {
    const data = await api(`/api/property/${btn.dataset.summary}/summary`);
    box.textContent = data.summary;
    box.hidden = false;
    btn.textContent = "Hide summary";
  } catch (err) {
    box.textContent = err.message; box.hidden = false; btn.textContent = "Try again";
  } finally {
    btn.disabled = false;
  }
});
