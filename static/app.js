// Stock Price Predictor Frontend Application (Indian Stocks & INR Supported)

let chartInstance = null;
let currentData = null;
let currencySymbol = "₹";

// Dataset visibility state
const visibilityState = {
  sma20: true,
  sma50: true,
  backtest: false,
  volume: false,
};

// Model names mapping
const modelLabels = {
  linear_regression: "Linear Regression",
  ridge: "Ridge Regression",
  random_forest: "Random Forest Regressor",
  gradient_boosting: "Gradient Boosting",
};

// Indian currency number formatter
function formatCurrency(val, symbol = currencySymbol) {
  if (val === null || val === undefined || isNaN(val)) return `${symbol}0.00`;
  return `${symbol}${Number(val).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

// Initialize application on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  initPopularStocks();
  initEventListeners();
  // Auto-run prediction on initial load with RELIANCE
  triggerPrediction();
});

function selectChip(symbol) {
  const input = document.getElementById("tickerInput");
  if (input) {
    input.value = symbol;
    triggerPrediction();
  }
}

function initPopularStocks() {
  const container = document.getElementById("popularChips");
  if (!container) return;
  fetch("/api/stocks/popular")
    .then((res) => res.json())
    .then((data) => {
      if (data && data.stocks && data.stocks.length > 0) {
        container.innerHTML = "";
        data.stocks.forEach((stock) => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "px-2.5 py-1 text-xs font-mono rounded-lg bg-slate-800/80 hover:bg-indigo-600/30 text-slate-300 hover:text-white border border-slate-700/60 hover:border-indigo-500/50 transition cursor-pointer flex items-center gap-1.5";
          btn.innerHTML = `<span class="font-bold">${stock.symbol}</span><span class="text-[10px] text-slate-400">${stock.category}</span>`;
          btn.onclick = () => selectChip(stock.symbol);
          container.appendChild(btn);
        });
      }
    })
    .catch((err) => {
      console.warn("Could not load popular stocks:", err);
    });
}

function initEventListeners() {
  const btn = document.getElementById("predictBtn");
  if (btn) btn.addEventListener("click", triggerPrediction);
  
  const input = document.getElementById("tickerInput");
  if (input) {
    input.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        triggerPrediction();
      }
    });
  }

  const modelSelect = document.getElementById("modelSelect");
  if (modelSelect) modelSelect.addEventListener("change", triggerPrediction);

  const periodSelect = document.getElementById("periodSelect");
  if (periodSelect) periodSelect.addEventListener("change", triggerPrediction);

  const horizonSelect = document.getElementById("horizonSelect");
  if (horizonSelect) horizonSelect.addEventListener("change", triggerPrediction);
}

function showLoading(isLoading) {
  const overlay = document.getElementById("chartLoadingOverlay");
  const btn = document.getElementById("predictBtn");
  const btnText = document.getElementById("btnText");
  const btnIcon = document.getElementById("btnIcon");

  if (overlay) {
    if (isLoading) {
      overlay.classList.remove("hidden");
      overlay.classList.add("flex");
    } else {
      overlay.classList.add("hidden");
      overlay.classList.remove("flex");
    }
  }

  if (btn) {
    if (isLoading) {
      btn.disabled = true;
      btn.classList.add("opacity-75", "cursor-not-allowed");
      if (btnText) btnText.textContent = "Analyzing...";
      if (btnIcon) btnIcon.classList.add("animate-spin");
    } else {
      btn.disabled = false;
      btn.classList.remove("opacity-75", "cursor-not-allowed");
      if (btnText) btnText.textContent = "Predict";
      if (btnIcon) btnIcon.classList.remove("animate-spin");
    }
  }
}

function showError(msg) {
  const alert = document.getElementById("errorAlert");
  const msgEl = document.getElementById("errorMessage");
  if (msgEl) msgEl.textContent = msg;
  if (alert) alert.classList.remove("hidden");
}

function hideError() {
  const alert = document.getElementById("errorAlert");
  if (alert) alert.classList.add("hidden");
}

async function triggerPrediction() {
  hideError();
  const tickerInput = document.getElementById("tickerInput");
  const ticker = (tickerInput ? tickerInput.value.trim().toUpperCase() : "RELIANCE") || "RELIANCE";
  const modelType = document.getElementById("modelSelect") ? document.getElementById("modelSelect").value : "linear_regression";
  const period = document.getElementById("periodSelect") ? document.getElementById("periodSelect").value : "1y";
  const horizonEl = document.getElementById("horizonSelect");
  const forecastDays = horizonEl ? parseInt(horizonEl.value, 10) || 7 : 7;

  showLoading(true);

  try {
    const response = await fetch("/api/stock/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticker: ticker,
        model_type: modelType,
        forecast_days: forecastDays,
        period: period,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    currentData = data;
    currencySymbol = data.meta && data.meta.currency_symbol ? data.meta.currency_symbol : "₹";
    updateUI(data);
  } catch (error) {
    console.error("Prediction API Error:", error);
    showError(error.message || "Failed to generate prediction. Check symbol and try again.");
  } finally {
    showLoading(false);
  }
}

function updateUI(data) {
  const meta = data.meta || {};
  const result = data.result || {};
  const summary = result.summary || {};
  const metrics = result.metrics || {};
  const modelType = result.model_type || "linear_regression";
  const sym = meta.currency_symbol || currencySymbol || "₹";

  // Badges & Labels
  const stockBadge = document.getElementById("stockSymbolBadge");
  if (stockBadge) stockBadge.textContent = meta.symbol || "---";

  const companyName = document.getElementById("cardCompanyName");
  if (companyName) companyName.textContent = meta.name || meta.symbol || "";

  const daysBadge = document.getElementById("forecastDaysBadge");
  if (daysBadge) daysBadge.textContent = `+${summary.forecast_days || 7} Days`;

  const modelBadge = document.getElementById("modelLabelBadge");
  if (modelBadge) modelBadge.textContent = modelLabels[modelType] || modelType;

  // KPI 1: Current Price
  const currentPriceEl = document.getElementById("cardCurrentPrice");
  if (currentPriceEl) currentPriceEl.textContent = formatCurrency(summary.current_price || meta.current_price || 0, sym);

  // KPI 2: Target Price & Change
  const targetPrice = summary.forecast_price || 0;
  const changePct = summary.forecast_change_pct || 0;
  const forecastPriceEl = document.getElementById("cardForecastPrice");
  if (forecastPriceEl) forecastPriceEl.textContent = formatCurrency(targetPrice, sym);
  
  const changeEl = document.getElementById("cardForecastChange");
  if (changeEl) {
    const isUp = changePct >= 0;
    const priceDiff = targetPrice - (summary.current_price || 0);
    changeEl.className = `text-xs font-semibold font-mono mt-0.5 flex items-center gap-1 ${isUp ? "text-emerald-400" : "text-rose-400"}`;
    changeEl.innerHTML = `
      <span>${isUp ? "▲" : "▼"} ${Math.abs(changePct).toFixed(2)}%</span>
      <span class="text-slate-500 font-normal">(${isUp ? "+" : ""}${formatCurrency(priceDiff, sym)})</span>
    `;
  }

  // KPI 3: Trend Signal
  const signalBadge = document.getElementById("signalBadge");
  if (signalBadge) {
    const signal = summary.signal || "NEUTRAL";
    if (signal === "BULLISH") {
      signalBadge.className = "inline-flex items-center px-3 py-1.5 rounded-xl text-sm font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
      signalBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 mr-2 animate-pulse"></span> BULLISH`;
    } else if (signal === "BEARISH") {
      signalBadge.className = "inline-flex items-center px-3 py-1.5 rounded-xl text-sm font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30";
      signalBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400 mr-2 animate-pulse"></span> BEARISH`;
    } else {
      signalBadge.className = "inline-flex items-center px-3 py-1.5 rounded-xl text-sm font-bold bg-slate-800 text-slate-300 border border-slate-700";
      signalBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-400 mr-2"></span> NEUTRAL`;
    }
  }

  // KPI 4: R2 Score
  const r2El = document.getElementById("cardR2Score");
  if (r2El) {
    const r2Val = metrics.r2_score !== undefined ? (metrics.r2_score * 100).toFixed(1) : "--";
    r2El.textContent = `${r2Val}%`;
  }

  // KPI 5: MAE & Directional Accuracy
  const maeEl = document.getElementById("cardMAE");
  if (maeEl) maeEl.textContent = formatCurrency(metrics.mae || 0, sym);

  const mapeEl = document.getElementById("cardMAPE");
  if (mapeEl) mapeEl.textContent = `MAPE: ${(metrics.mape_pct || 0).toFixed(2)}% | RMSE: ${formatCurrency(metrics.rmse || 0, sym)}`;

  const dirEl = document.getElementById("cardDirectionalAcc");
  if (dirEl) dirEl.textContent = `Dir: ${(metrics.directional_accuracy_pct || 50).toFixed(0)}%`;

  // Render Chart
  renderChart(result, sym);

  // Render Forecast Table
  renderForecastTable(result.forecast, summary.current_price, sym);

  // Render Feature Drivers
  renderFeatureImportance(result.feature_importance);
}

function renderChart(result, sym) {
  const canvas = document.getElementById("stockChart");
  if (!canvas) return;

  const history = result.history || [];
  const forecast = result.forecast || [];
  const fittedTest = result.fitted_test || [];

  const histDates = history.map((d) => d.date);
  const forecastDates = forecast.map((d) => d.date);
  const allLabels = [...histDates, ...forecastDates];

  const actualPrices = history.map((d) => d.close);
  const paddedActual = [...actualPrices, ...forecast.map(() => null)];

  const sma20 = [...history.map((d) => d.sma_20), ...forecast.map(() => null)];
  const sma50 = [...history.map((d) => d.sma_50), ...forecast.map(() => null)];
  const volumeData = [...history.map((d) => d.volume), ...forecast.map(() => null)];

  const lastHistPrice = history.length > 0 ? history[history.length - 1].close : null;
  const forecastSeries = new Array(history.length - 1).fill(null);
  forecastSeries.push(lastHistPrice);
  forecast.forEach((f) => forecastSeries.push(f.predicted_close));

  const backtestMap = {};
  fittedTest.forEach((item) => {
    backtestMap[item.date] = item.predicted;
  });
  const backtestSeries = allLabels.map((date) => (backtestMap[date] !== undefined ? backtestMap[date] : null));

  if (typeof Chart === "undefined") {
    drawCanvasFallback(canvas, allLabels, actualPrices, forecastSeries, sym);
    return;
  }

  const ctx = canvas.getContext("2d");

  if (chartInstance) {
    chartInstance.destroy();
  }

  const gradient = ctx.createLinearGradient(0, 0, 0, 380);
  gradient.addColorStop(0, "rgba(99, 102, 241, 0.28)");
  gradient.addColorStop(1, "rgba(99, 102, 241, 0.0)");

  try {
    chartInstance = new Chart(ctx, {
      type: "line",
      data: {
        labels: allLabels,
        datasets: [
          {
            label: "Historical Close",
            data: paddedActual,
            borderColor: "#6366f1",
            backgroundColor: gradient,
            fill: true,
            tension: 0.15,
            borderWidth: 2.2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: "#6366f1",
            pointHoverBorderColor: "#ffffff",
            yAxisID: "y",
          },
          {
            label: "AI Forecast Trajectory",
            data: forecastSeries,
            borderColor: "#22d3ee",
            backgroundColor: "transparent",
            borderDash: [6, 4],
            borderWidth: 2.5,
            tension: 0.1,
            pointRadius: (ctx) => (ctx.dataIndex === forecastSeries.length - 1 ? 6 : 3),
            pointBackgroundColor: "#22d3ee",
            pointBorderColor: "#0b0f19",
            pointBorderWidth: 2,
            pointHoverRadius: 7,
            yAxisID: "y",
          },
          {
            label: "SMA 20",
            data: sma20,
            borderColor: "#f59e0b",
            borderWidth: 1.5,
            borderDash: [3, 3],
            pointRadius: 0,
            fill: false,
            hidden: !visibilityState.sma20,
            yAxisID: "y",
          },
          {
            label: "SMA 50",
            data: sma50,
            borderColor: "#a855f7",
            borderWidth: 1.5,
            borderDash: [3, 3],
            pointRadius: 0,
            fill: false,
            hidden: !visibilityState.sma50,
            yAxisID: "y",
          },
          {
            label: "Backtest Model Fit",
            data: backtestSeries,
            borderColor: "#10b981",
            borderWidth: 1.8,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false,
            hidden: !visibilityState.backtest,
            yAxisID: "y",
          },
          {
            type: "bar",
            label: "Trading Volume",
            data: volumeData,
            backgroundColor: "rgba(100, 116, 139, 0.18)",
            hoverBackgroundColor: "rgba(100, 116, 139, 0.35)",
            yAxisID: "yVolume",
            hidden: !visibilityState.volume,
            order: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            backgroundColor: "rgba(15, 23, 42, 0.95)",
            titleColor: "#94a3b8",
            bodyColor: "#f8fafc",
            borderColor: "#334155",
            borderWidth: 1,
            padding: 12,
            boxPadding: 4,
            usePointStyle: true,
            callbacks: {
              label: function (context) {
                if (context.dataset.yAxisID === "yVolume") {
                  return ` Volume: ${(context.raw || 0).toLocaleString("en-IN")}`;
                }
                return ` ${context.dataset.label}: ${formatCurrency(context.raw || 0, sym)}`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: {
              color: "rgba(30, 41, 59, 0.6)",
              drawBorder: false,
            },
            ticks: {
              color: "#64748b",
              maxTicksLimit: 12,
              font: { family: "'JetBrains Mono', monospace", size: 10 },
            },
          },
          y: {
            position: "right",
            grid: {
              color: "rgba(30, 41, 59, 0.6)",
              drawBorder: false,
            },
            ticks: {
              color: "#94a3b8",
              font: { family: "'JetBrains Mono', monospace", size: 11 },
              callback: (val) => `${sym}${Number(val).toLocaleString("en-IN")}`,
            },
          },
          yVolume: {
            position: "left",
            display: visibilityState.volume,
            grid: { drawOnChartArea: false },
            ticks: {
              color: "#475569",
              font: { size: 9 },
              callback: (val) => (val / 1e5).toFixed(1) + "L",
            },
          },
        },
      },
    });
  } catch (err) {
    console.error("Error creating Chart.js instance, drawing fallback canvas:", err);
    drawCanvasFallback(canvas, allLabels, actualPrices, forecastSeries, sym);
  }
}

function drawCanvasFallback(canvas, labels, histPrices, forecastPrices, sym) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.parentElement.clientWidth || 800;
  const h = canvas.height = 400;

  ctx.fillStyle = "#0c1220";
  ctx.fillRect(0, 0, w, h);

  const validVals = [...histPrices.filter(v => v !== null), ...forecastPrices.filter(v => v !== null)];
  if (validVals.length === 0) return;

  const min = Math.min(...validVals) * 0.98;
  const max = Math.max(...validVals) * 1.02;
  const totalPoints = labels.length;

  const getX = (i) => 40 + (i / (totalPoints - 1)) * (w - 80);
  const getY = (v) => h - 40 - ((v - min) / (max - min)) * (h - 80);

  // Draw grid
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = 40 + (i / 4) * (h - 80);
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 40, y);
    ctx.stroke();
    const val = max - (i / 4) * (max - min);
    ctx.fillStyle = "#64748b";
    ctx.font = "10px monospace";
    ctx.fillText(`${sym}${val.toFixed(1)}`, w - 45, y + 3);
  }

  // Draw historical line
  ctx.strokeStyle = "#6366f1";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  let first = true;
  for (let i = 0; i < histPrices.length; i++) {
    if (histPrices[i] !== null) {
      const x = getX(i);
      const y = getY(histPrices[i]);
      if (first) { ctx.moveTo(x, y); first = false; }
      else { ctx.lineTo(x, y); }
    }
  }
  ctx.stroke();

  // Draw forecast line
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 3;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  first = true;
  for (let i = 0; i < forecastPrices.length; i++) {
    if (forecastPrices[i] !== null) {
      const x = getX(i);
      const y = getY(forecastPrices[i]);
      if (first) { ctx.moveTo(x, y); first = false; }
      else { ctx.lineTo(x, y); }
    }
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function toggleDataset(type) {
  if (!chartInstance) return;
  visibilityState[type] = !visibilityState[type];

  const btnMap = {
    sma20: "toggleSMA20",
    sma50: "toggleSMA50",
    backtest: "toggleBacktest",
    volume: "toggleVolume",
  };

  const btn = document.getElementById(btnMap[type]);
  if (btn) {
    if (visibilityState[type]) {
      btn.classList.add("ring-2", "ring-indigo-500/50");
    } else {
      btn.classList.remove("ring-2", "ring-indigo-500/50");
    }
  }

  const datasetIdxMap = {
    sma20: 2,
    sma50: 3,
    backtest: 4,
    volume: 5,
  };

  const idx = datasetIdxMap[type];
  if (idx !== undefined && chartInstance.data.datasets[idx]) {
    chartInstance.data.datasets[idx].hidden = !visibilityState[type];
    if (type === "volume") {
      chartInstance.options.scales.yVolume.display = visibilityState.volume;
    }
    chartInstance.update();
  }
}

function renderForecastTable(forecastList, basePrice, sym) {
  const tbody = document.getElementById("forecastTableBody");
  if (!tbody) return;
  if (!forecastList || forecastList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" class="py-4 text-center text-slate-500">No forecast data</td></tr>`;
    return;
  }

  let prevPrice = basePrice;
  let html = "";

  forecastList.forEach((item) => {
    const predPrice = item.predicted_close;
    const dailyDiff = predPrice - prevPrice;
    const dailyPct = (dailyDiff / (prevPrice || 1)) * 100;
    const cumPct = ((predPrice - basePrice) / (basePrice || 1)) * 100;
    const isUp = dailyDiff >= 0;
    const isCumUp = cumPct >= 0;

    html += `
      <tr class="hover:bg-slate-800/40 transition">
        <td class="py-2 px-3 text-slate-300 font-semibold">${item.date}</td>
        <td class="py-2 px-3 text-cyan-300 font-bold">${formatCurrency(predPrice, sym)}</td>
        <td class="py-2 px-3 ${isUp ? "text-emerald-400" : "text-rose-400"}">
          ${isUp ? "+" : ""}${formatCurrency(dailyDiff, sym)} (${isUp ? "+" : ""}${dailyPct.toFixed(2)}%)
        </td>
        <td class="py-2 px-3 text-right font-bold ${isCumUp ? "text-emerald-400" : "text-rose-400"}">
          ${isCumUp ? "+" : ""}${cumPct.toFixed(2)}%
        </td>
      </tr>
    `;
    prevPrice = predPrice;
  });

  tbody.innerHTML = html;
}

function renderFeatureImportance(importanceMap) {
  const container = document.getElementById("featureImportanceContainer");
  if (!container) return;
  if (!importanceMap || Object.keys(importanceMap).length === 0) {
    container.innerHTML = `<p class="text-xs text-slate-500">Feature weights not available for this model.</p>`;
    return;
  }

  const entries = Object.entries(importanceMap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxVal = Math.max(...entries.map((e) => Math.abs(e[1])), 0.0001);

  let html = "";
  entries.slice(0, 8).forEach(([feature, val]) => {
    const pct = Math.min(100, Math.round((Math.abs(val) / maxVal) * 100));
    const isPositive = val >= 0;

    html += `
      <div class="space-y-1">
        <div class="flex justify-between text-xs font-mono">
          <span class="text-slate-300 font-medium">${feature}</span>
          <span class="${isPositive ? "text-indigo-300" : "text-amber-400"}">${val > 0 ? "+" : ""}${val}</span>
        </div>
        <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
          <div class="h-full ${isPositive ? "bg-indigo-500" : "bg-amber-500"}" style="width: ${pct}%"></div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}
