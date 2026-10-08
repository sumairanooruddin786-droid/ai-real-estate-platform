const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatBtn = document.getElementById("chat-btn");
const startersEl = document.getElementById("starters");

let mode = "advisor";
const history = [];

const STARTERS = {
  advisor: ["Which neighborhoods are the most expensive?", "Is College Creek a good place to buy?",
            "How much should I save for closing costs?", "What makes a home price go up?"],
  docs: ["What contingencies should an offer include?", "How is cap rate calculated?",
         "What legal checks should I do before buying?", "What does PMI mean?"],
};
const NOTES = {
  advisor: "Uses neighborhood statistics from the sales data, plus the documents when they help.",
  docs: "Answers only from the documents in the knowledge base and cites each source.",
};

function addMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  wrap.innerHTML = role === "bot" ? `<div class="bubble">${html}</div>` : esc(html);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function sourcesHTML(sources) {
  if (!sources || !sources.length) return "";
  return `<details class="sources"><summary>${sources.length} source${sources.length > 1 ? "s" : ""}</summary><ul>` +
    sources.map((s) => `<li><b>[${s.n}] ${esc(s.source)}</b>, page ${s.page}<br>${esc(s.snippet)}</li>`).join("") +
    "</ul></details>";
}

function showStarters() {
  startersEl.innerHTML = STARTERS[mode].map((q) => `<button type="button">${esc(q)}</button>`).join("");
}

function setMode(next) {
  mode = next;
  document.getElementById("mode-advisor").setAttribute("aria-pressed", next === "advisor");
  document.getElementById("mode-docs").setAttribute("aria-pressed", next === "docs");
  document.getElementById("mode-note").textContent = NOTES[next];
  history.length = 0;
  showStarters();
}

async function ask(question) {
  question = question.trim();
  if (!question) { chatInput.focus(); return; }
  addMessage("user", question);
  startersEl.innerHTML = "";
  chatInput.value = "";
  chatBtn.disabled = true;
  const pending = addMessage("bot", '<span class="typing">Looking through the data...</span>');
  try {
    const r = await postJSON("/api/chat", { message: question, mode, history: history.slice(-6) });
    pending.querySelector(".bubble").innerHTML = renderText(r.answer) + sourcesHTML(r.sources);
    history.push({ role: "user", content: question }, { role: "assistant", content: r.answer });
  } catch (err) {
    pending.querySelector(".bubble").innerHTML = `<span style="color:var(--danger)">${esc(err.message)}</span>`;
  } finally {
    chatBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
    chatInput.focus();
  }
}

chatForm.addEventListener("submit", (e) => { e.preventDefault(); ask(chatInput.value); });
startersEl.addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") ask(e.target.textContent); });
document.getElementById("mode-advisor").addEventListener("click", () => setMode("advisor"));
document.getElementById("mode-docs").addEventListener("click", () => setMode("docs"));

// ---- knowledge base panel
const docList = document.getElementById("doc-list");
const uploadMsg = document.getElementById("upload-msg");

async function loadDocs() {
  try {
    const { documents } = await api("/api/documents");
    docList.innerHTML = documents.length
      ? documents.map((d) => `<li><span>${esc(d.name)} <span class="n">${d.chunks} passages</span></span>` +
          (d.uploaded ? `<button type="button" data-remove="${esc(d.name)}">Remove</button>` : "") + "</li>").join("")
      : '<li class="muted">No documents yet. Upload one to get started.</li>';
  } catch (err) {
    docList.innerHTML = `<li class="muted">${esc(err.message)}</li>`;
  }
}

document.getElementById("doc-upload").addEventListener("click", async () => {
  const file = document.getElementById("doc-file").files[0];
  if (!file) { uploadMsg.textContent = "Choose a PDF, TXT or MD file first."; return; }
  const data = new FormData();
  data.append("file", file);
  uploadMsg.textContent = "Indexing...";
  try {
    const r = await api("/api/documents", { method: "POST", body: data });
    uploadMsg.textContent = `${r.name} added (${r.chunks} passages). You can ask about it now.`;
    document.getElementById("doc-file").value = "";
    loadDocs();
  } catch (err) {
    uploadMsg.textContent = err.message;
  }
});

docList.addEventListener("click", async (e) => {
  const name = e.target.dataset.remove;
  if (!name) return;
  try { await api("/api/documents/" + encodeURIComponent(name), { method: "DELETE" }); loadDocs(); }
  catch (err) { uploadMsg.textContent = err.message; }
});

addMessage("bot", renderText("Hi, I'm Plotwise. Ask me about Ames home prices and neighborhoods, or how buying and investing work. Pick a question below to start."));
showStarters();
loadDocs();
