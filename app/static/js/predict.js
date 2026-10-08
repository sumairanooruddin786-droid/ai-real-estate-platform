const form = document.getElementById("predict-form");
const slip = document.getElementById("slip");
const errBox = document.getElementById("slip-error");
const NUMERIC = ["OverallQual", "GrLivArea", "TotalBsmtSF", "LotArea", "BedroomAbvGr", "FullBath",
                 "TotRmsAbvGrd", "GarageCars", "Fireplaces", "YearBuilt", "YearRemodAdd"];
let timer = null, latest = 0;

function readForm() {
  const body = {};
  for (const el of form.elements) {
    if (!el.name || el.value === "") continue;
    body[el.name] = NUMERIC.includes(el.name) ? Number(el.value) : el.value;
  }
  return body;
}

function render(r) {
  document.getElementById("slip-price").textContent = money(r.price);
  document.getElementById("slip-range").textContent = `Likely range ${money(r.low)} to ${money(r.high)}`;
  const cmp = document.getElementById("slip-compare");
  if (r.neighborhood_median) {
    const up = r.vs_neighborhood_pct >= 0;
    cmp.innerHTML = `The median home in ${esc(r.neighborhood_name)} sold for ${money(r.neighborhood_median)}. ` +
      `This one is <span class="${up ? "up" : "down"}">${Math.abs(r.vs_neighborhood_pct)}% ${up ? "above" : "below"}</span> that.`;
  }
  document.getElementById("slip-whatif").innerHTML = r.what_if
    .map((w) => `<li><span>${esc(w.change)}</span><span class="delta ${w.delta >= 0 ? "up" : "down"}">${signedMoney(w.delta)}</span></li>`)
    .join("");
  document.getElementById("slip-model").textContent =
    `Estimated with a ${r.model} model. The range reflects how far off it was on homes it had not seen.`;
}

async function update() {
  const id = ++latest;
  slip.classList.add("loading");
  try {
    const r = await postJSON("/api/predict", readForm());
    if (id !== latest) return;          // a newer request is already on its way
    errBox.hidden = true;
    render(r);
    slip.hidden = false;
  } catch (e) {
    if (id !== latest) return;
    errBox.textContent = e.message;
    errBox.hidden = false;
  } finally {
    if (id === latest) slip.classList.remove("loading");
  }
}

form.addEventListener("input", (e) => {
  if (e.target.id === "OverallQual") document.getElementById("qual-out").textContent = e.target.value;
  clearTimeout(timer);
  timer = setTimeout(update, 350);
});
form.addEventListener("submit", (e) => e.preventDefault());
update();
