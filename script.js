// -----------------------------------------------
// script.js — Connects Frontend UI to Flask + MySQL
// -----------------------------------------------

let currentPredictionId = null;
let selectedStars = 5;
let cachedHistoryRows = [];

function showPage(pageId) {
  document.querySelectorAll(".page").forEach(p => p.classList.add("hidden"));
  document.getElementById(pageId).classList.remove("hidden");
}

function goToResult() {
  const text = document.getElementById("newsInput").value.trim();

  if (!text) {
    shakeElement(document.querySelector(".left-side"));
    return;
  }

  showPage("result-page");

  document.getElementById("spinner").style.display = "block";
  document.getElementById("resultInner").style.display = "none";
  document.getElementById("confFill").style.width = "0%";

  sendToFlask(text);
}

function goHome() {
  showPage("home-page");
  loadStats();
}

function goToHistory() {
  showPage("history-page");
  loadHistory();
}

function resetInput() {
  document.getElementById("newsInput").value = "";
  document.getElementById("newsInput").focus();
}

// -----------------------------------------------
// POST /predict — Saves or fetches from MySQL
// -----------------------------------------------
async function sendToFlask(newsText) {
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: newsText })
    });

    const result = await response.json();
    if (!response.ok) {
      showError(result.error || "Server error. Check the VS Code terminal.");
      return;
    }

    currentPredictionId = result.id;
    showResult(result);
  } catch (err) {
    showError("Could not reach Flask. Make sure 'python app.py' is running.");
    console.error(err);
  }
}

function showResult(data) {
  document.getElementById("spinner").style.display = "none";

  const isReal = data.verdict === "REAL";
  const conf = Math.min(97, Math.max(50, Number(data.confidence) || 75));

  const badge = document.getElementById("verdictBadge");
  badge.className = "verdict-badge " + (isReal ? "real" : "fake");
  badge.textContent = isReal ? "✅  REAL NEWS" : "❌  FAKE NEWS";

  const cacheNote = data.is_cached
    ? ` (Cached in MySQL • Checked ${data.check_count} times)`
    : ` (Saved to MySQL as ID #${data.id})`;

  document.getElementById("resultReason").textContent = (data.reason || "") + cacheNote;

  const fill = document.getElementById("confFill");
  fill.style.background = isReal ? "#46b982" : "#cd465a";
  fill.style.width = "0%";

  document.getElementById("confNumber").textContent = conf + "%";
  document.getElementById("resultInner").style.display = "flex";

  setTimeout(() => {
    fill.style.width = conf + "%";
  }, 80);
}

function showError(message) {
  document.getElementById("spinner").style.display = "none";
  const badge = document.getElementById("verdictBadge");
  badge.className = "verdict-badge error";
  badge.textContent = "⚠️  Something went wrong";
  document.getElementById("resultReason").textContent = message;
  document.getElementById("confNumber").textContent = "";
  document.getElementById("resultInner").style.display = "flex";
}

// -----------------------------------------------
// GET /history — Render MySQL Predictions Table
// -----------------------------------------------
async function loadHistory() {
  const tbody = document.getElementById("historyTableBody");
  if (!tbody) return;

  const qInput = document.getElementById("historySearch");
  const fSelect = document.getElementById("historyFilter");
  const q = qInput ? encodeURIComponent(qInput.value.trim()) : "";
  const verdict = fSelect ? fSelect.value : "";

  try {
    const res = await fetch(`/history?q=${q}&verdict=${verdict}`);
    if (!res.ok) return;
    cachedHistoryRows = await res.json();

    if (cachedHistoryRows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="empty-cell">No records found in MySQL.</td></tr>`;
      return;
    }

    tbody.innerHTML = cachedHistoryRows.map((row, idx) => `
      <tr>
        <td>#${row.id}</td>
        <td title="${row.news_text.replace(/"/g, "&quot;")}">
          ${row.news_text.length > 52 ? row.news_text.slice(0, 52) + "…" : row.news_text}
        </td>
        <td class="${row.verdict === "REAL" ? "verdict-real-text" : "verdict-fake-text"}">
          ${row.verdict}
        </td>
        <td>${row.confidence}%</td>
        <td>${row.check_count}x</td>
        <td>${row.created_at}</td>
        <td>
          <button class="btn-mini" onclick="reopenHistoryItem(${idx})">View</button>
          <button class="btn-mini danger" onclick="deleteHistoryItem(${row.id})">✕</button>
        </td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

function reopenHistoryItem(index) {
  const item = cachedHistoryRows[index];
  if (!item) return;
  document.getElementById("newsInput").value = item.news_text;
  goToResult();
}

async function deleteHistoryItem(id) {
  try {
    await fetch(`/history/${id}`, { method: "DELETE" });
    loadHistory();
    loadStats();
  } catch (err) {
    console.error("Failed to delete item:", err);
  }
}

// -----------------------------------------------
// GET /stats — Update Home Page MySQL Strip
// -----------------------------------------------
async function loadStats() {
  try {
    const res = await fetch("/stats");
    if (!res.ok) return;
    const s = await res.json();
    document.getElementById("statUnique").textContent = s.total_unique;
    document.getElementById("statCache").textContent = s.cache_hits;
    document.getElementById("statSplit").textContent = `${s.real_count} / ${s.fake_count}`;
    document.getElementById("statAcc").textContent = `${s.agreement_rate}%`;
  } catch (err) {
    console.error("Stats fetch skipped:", err);
  }
}

// -----------------------------------------------
// POST /feedback — Save user feedback to MySQL
// -----------------------------------------------
async function sendFeedback(isAccurate) {
  if (!currentPredictionId) return;
  try {
    const res = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_id: currentPredictionId,
        is_accurate: isAccurate
      })
    });
    const data = await res.json();
    if (res.ok) {
      document.getElementById("resultReason").textContent =
        `Thank you! Feedback saved to MySQL (Label: ${data.suggested_label}).`;
      loadStats();
    }
  } catch (err) {
    console.error("Feedback error:", err);
  }
}

// -----------------------------------------------
// Rating Modal & POST /rate
// -----------------------------------------------
function openRateModal() {
  document.getElementById("rateFeedbackMsg").textContent = "";
  document.getElementById("rateModal").classList.remove("hidden");
}

function closeRateModal() {
  document.getElementById("rateModal").classList.remove("hidden");
  document.getElementById("rateModal").classList.add("hidden");
}

function selectStars(count) {
  selectedStars = count;
  document.querySelectorAll("#starSelector .star").forEach((star, i) => {
    star.classList.toggle("active", i < count);
  });
}

async function handleRatingSubmit() {
  const comment = document.getElementById("rateComment").value.trim();
  await submitRating(selectedStars, comment);
}

async function submitRating(stars, comment = "") {
  try {
    const res = await fetch("/rate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stars, comment })
    });
    const data = await res.json();
    if (res.ok) {
      document.querySelectorAll(".rate-btn").forEach(btn => {
        btn.textContent = `⭐ Rate Us (${data.average} ★ • ${data.total_votes})`;
      });
      document.getElementById("rateFeedbackMsg").textContent =
        `${data.message} Community Average: ${data.average} ★ (${data.total_votes} votes)`;
      document.getElementById("ratingSubtitle").textContent =
        `Current Average: ${data.average} / 5 across ${data.total_votes} ratings`;
    }
  } catch (err) {
    console.error("Rating error:", err);
  }
}

function shakeElement(el) {
  el.animate(
    [
      { transform: "translateX(0)" },
      { transform: "translateX(-8px)" },
      { transform: "translateX(8px)" },
      { transform: "translateX(-5px)" },
      { transform: "translateX(5px)" },
      { transform: "translateX(0)" }
    ],
    { duration: 380, easing: "ease-in-out" }
  );
}

document.addEventListener("keydown", function (e) {
  if (e.ctrlKey && e.key === "Enter") {
    goToResult();
  }
});

window.addEventListener("DOMContentLoaded", () => {
  loadStats();
  loadHistory();
});