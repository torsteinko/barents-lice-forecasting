const METRICS = {
  near_term_risk: {
    label: "Near-term breach risk",
    kind: "probability",
    thresholds: [0.35, 0.55, 0.75],
    description: "Maximum of the 1-week and 2-week breach probabilities.",
  },
  classifier_12w_score: {
    label: "12-week breach risk",
    kind: "probability",
    thresholds: [0.4, 0.6, 0.8],
    description: "Longer-horizon outbreak pressure.",
  },
  classifier_2w_score: {
    label: "2-week breach risk",
    kind: "probability",
    thresholds: [0.35, 0.55, 0.75],
    description: "Classifier probability for the next two weeks.",
  },
  classifier_1w_score: {
    label: "1-week breach risk",
    kind: "probability",
    thresholds: [0.35, 0.55, 0.8],
    description: "Immediate pressure for the next reported week.",
  },
  femaleadult_to_limit_ratio: {
    label: "Current limit ratio",
    kind: "ratio",
    thresholds: [0.5, 1.0, 1.5],
    description: "Current female-adult lice relative to the weekly threshold.",
  },
};

const RISK_FILTERS = ["all", "critical", "high", "watch", "stable"];
const CHAT_SUGGESTIONS = [
  "Which site in Vestland has the biggest chance of outbreak the next 2 weeks?",
  "Which visible sites are already above the lice limit right now?",
  "Show me the highest 12-week risk sites that were treated recently.",
];

const state = {
  features: [],
  visible: [],
  selectedId: null,
  datasetMeta: {},
  currentMetric: "near_term_risk",
  riskFilter: "all",
  area: "all",
  county: "all",
  query: "",
  overLimitOnly: false,
  recentTreatmentOnly: false,
  countedOnly: false,
  chatOpen: false,
};

const map = new maplibregl.Map({
  container: "map",
  style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  center: [14.2, 65.1],
  zoom: 4.25,
  pitch: 0,
  bearing: 0,
});

map.addControl(new maplibregl.NavigationControl(), "top-right");

const hoverPopup = new maplibregl.Popup({
  closeButton: false,
  closeOnClick: false,
  offset: 14,
});

function formatStatusLabel(value) {
  const text = formatText(value);
  if (text === "--") {
    return text;
  }
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatNumber(value, digits = 2) {
  const numeric = toNumber(value);
  if (numeric === null) {
    return "--";
  }
  return numeric.toFixed(digits);
}

function formatInteger(value) {
  const numeric = toNumber(value);
  if (numeric === null) {
    return "--";
  }
  return `${Math.round(numeric)}`;
}

function formatPercent(value) {
  const numeric = toNumber(value);
  if (numeric === null) {
    return "--";
  }
  return `${Math.round(numeric * 100)}%`;
}

function formatDate(value) {
  if (!value) {
    return "--";
  }
  return String(value);
}

function formatCaseDate(value) {
  if (!value) {
    return "--";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value);
  }
  return parsed.toISOString().slice(0, 10);
}

function formatText(value) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  return String(value);
}

function formatBoolean(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  return value ? "Yes" : "No";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderMarkdown(text) {
  const safeText = formatText(text);
  if (window.marked && window.DOMPurify) {
    const rawHtml = window.marked.parse(safeText, {
      breaks: true,
      gfm: true,
    });
    return window.DOMPurify.sanitize(rawHtml);
  }
  return escapeHtml(safeText).replace(/\n/g, "<br>");
}

function metricValue(site, metricKey = state.currentMetric) {
  return toNumber(site[metricKey]);
}

function formatMetric(site, metricKey = state.currentMetric) {
  const metric = METRICS[metricKey];
  const value = metricValue(site, metricKey);
  if (value === null) {
    return "--";
  }
  return metric.kind === "probability" ? formatPercent(value) : `${formatNumber(value)}x`;
}

function riskBucket(value, metricKey = state.currentMetric) {
  const metric = METRICS[metricKey];
  const numeric = toNumber(value);
  if (numeric === null) {
    return "stable";
  }
  if (numeric >= metric.thresholds[2]) {
    return "critical";
  }
  if (numeric >= metric.thresholds[1]) {
    return "high";
  }
  if (numeric >= metric.thresholds[0]) {
    return "watch";
  }
  return "stable";
}

function markerRadius(value, metricKey = state.currentMetric) {
  const numeric = toNumber(value);
  if (numeric === null) {
    return 6;
  }
  return METRICS[metricKey].kind === "probability"
    ? 7 + Math.min(12, numeric * 16)
    : 7 + Math.min(12, numeric * 4.2);
}

function bucketColor(bucket) {
  if (bucket === "critical") {
    return "#c54032";
  }
  if (bucket === "high") {
    return "#d47a28";
  }
  if (bucket === "watch") {
    return "#1f867d";
  }
  return "#74828a";
}

function normalizeFeature(feature) {
  const properties = feature.properties || {};
  const geometry = feature.geometry || { coordinates: [null, null] };
  const nearTermValues = [
    toNumber(properties.classifier_1w_score),
    toNumber(properties.classifier_2w_score),
  ].filter((value) => value !== null);
  const nearTermCountValues = [
    toNumber(properties.count_1w_prediction),
    toNumber(properties.count_2w_prediction),
  ].filter((value) => value !== null);

  return {
    ...properties,
    id: String(properties.sitenumber),
    longitude: geometry.coordinates[0],
    latitude: geometry.coordinates[1],
    near_term_risk: nearTermValues.length ? Math.max(...nearTermValues) : null,
    near_term_count_prediction: nearTermCountValues.length ? Math.max(...nearTermCountValues) : null,
  };
}

async function fetchSiteDataset() {
  const endpoints = ["/api/sites", "../results/site_map.geojson"];
  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint);
      if (!response.ok) {
        continue;
      }
      return await response.json();
    } catch (error) {
      console.warn(`Failed to load ${endpoint}`, error);
    }
  }
  throw new Error("Could not load site data from the API or the saved GeoJSON artifact.");
}

function renderCaseDate() {
  const caseDate = state.datasetMeta?.case_cutoff_date || null;
  document.getElementById("case-date-value").textContent = formatCaseDate(caseDate);
  document.getElementById("case-date-note").textContent = caseDate
    ? `Everything shown here is capped at ${formatCaseDate(caseDate)}, not the real present day.`
    : "All site status, treatment context, and answers are capped at the case cutoff.";
}

function renderMetricSwitcher() {
  const container = document.getElementById("metric-switcher");
  container.innerHTML = Object.entries(METRICS)
    .map(
      ([metricKey, metric]) => `
        <button type="button" class="metric-chip ${metricKey === state.currentMetric ? "active" : ""}" data-metric="${metricKey}">
          <span>${escapeHtml(metric.label)}</span>
        </button>
      `,
    )
    .join("");

  container.querySelectorAll("[data-metric]").forEach((button) => {
    button.addEventListener("click", () => {
      state.currentMetric = button.dataset.metric;
      refreshView();
    });
  });

  document.getElementById("metric-title").textContent = METRICS[state.currentMetric].label;
}

function renderRiskFilter() {
  const container = document.getElementById("risk-filter-row");
  container.innerHTML = RISK_FILTERS.map(
    (risk) => `
      <button type="button" class="risk-chip ${state.riskFilter === risk ? "active" : ""}" data-risk="${risk}">
        ${escapeHtml(risk === "all" ? "All" : formatStatusLabel(risk))}
      </button>
    `,
  ).join("");

  container.querySelectorAll("[data-risk]").forEach((button) => {
    button.addEventListener("click", () => {
      state.riskFilter = button.dataset.risk;
      refreshView();
    });
  });
}

function populateFilters() {
  const areaFilter = document.getElementById("area-filter");
  const countyFilter = document.getElementById("county-filter");
  const areas = [...new Set(state.features.map((site) => site.productionarea).filter(Boolean))].sort();
  const counties = [...new Set(state.features.map((site) => site.county).filter(Boolean))].sort();

  areaFilter.innerHTML = ['<option value="all">All production areas</option>']
    .concat(areas.map((area) => `<option value="${escapeHtml(area)}">${escapeHtml(area)}</option>`))
    .join("");
  countyFilter.innerHTML = ['<option value="all">All counties</option>']
    .concat(counties.map((county) => `<option value="${escapeHtml(county)}">${escapeHtml(county)}</option>`))
    .join("");
}

function buildLegend() {
  const metric = METRICS[state.currentMetric];
  const bands = [
    { label: `Below ${formatLegendValue(metric.thresholds[0], metric.kind)}`, bucket: "stable" },
    { label: `${formatLegendValue(metric.thresholds[0], metric.kind)} to ${formatLegendValue(metric.thresholds[1], metric.kind)}`, bucket: "watch" },
    { label: `${formatLegendValue(metric.thresholds[1], metric.kind)} to ${formatLegendValue(metric.thresholds[2], metric.kind)}`, bucket: "high" },
    { label: `Above ${formatLegendValue(metric.thresholds[2], metric.kind)}`, bucket: "critical" },
  ];

  document.getElementById("legend").innerHTML = `
    <div class="block-header">
      <h3>Risk Bands</h3>
      <span>${escapeHtml(metric.description)}</span>
    </div>
    ${bands
      .map(
        (band) => `
          <div class="legend-row">
            <span>${escapeHtml(`${formatStatusLabel(band.bucket)} · ${band.label}`)}</span>
            <span class="legend-swatch" style="background:${bucketColor(band.bucket)}"></span>
          </div>
        `,
      )
      .join("")}
  `;
}

function formatLegendValue(value, kind) {
  return kind === "probability" ? formatPercent(value) : `${formatNumber(value)}x`;
}

function filterSites() {
  const query = state.query.trim().toLowerCase();
  return state.features.filter((site) => {
    const siteRisk = riskBucket(metricValue(site));
    const matchesRisk = state.riskFilter === "all" || siteRisk === state.riskFilter;
    const matchesArea = state.area === "all" || site.productionarea === state.area;
    const matchesCounty = state.county === "all" || site.county === state.county;
    const matchesSearch =
      query === "" ||
      formatText(site.sitename).toLowerCase().includes(query) ||
      formatText(site.municipality).toLowerCase().includes(query) ||
      formatText(site.sitenumber).toLowerCase().includes(query);
    const matchesOverLimit = !state.overLimitOnly || Boolean(site.currently_over_limit);
    const matchesRecentTreatment =
      !state.recentTreatmentOnly ||
      ((toNumber(site.weeks_since_any_treatment) ?? Number.POSITIVE_INFINITY) <= 4);
    const matchesCounted = !state.countedOnly || Boolean(site.havecountedlice);

    return (
      matchesRisk &&
      matchesArea &&
      matchesCounty &&
      matchesSearch &&
      matchesOverLimit &&
      matchesRecentTreatment &&
      matchesCounted
    );
  });
}

function sortSites(sites) {
  return [...sites].sort((left, right) => {
    const primaryGap = (metricValue(right) ?? -1) - (metricValue(left) ?? -1);
    if (primaryGap !== 0) {
      return primaryGap;
    }
    const ratioGap = (toNumber(right.femaleadult_to_limit_ratio) ?? -1) - (toNumber(left.femaleadult_to_limit_ratio) ?? -1);
    if (ratioGap !== 0) {
      return ratioGap;
    }
    return formatText(left.sitename).localeCompare(formatText(right.sitename));
  });
}

function renderStats() {
  const visible = state.visible;
  const critical = visible.filter((site) => riskBucket(metricValue(site)) === "critical").length;
  const overLimit = visible.filter((site) => Boolean(site.currently_over_limit)).length;
  const treated = visible.filter((site) => (toNumber(site.weeks_since_any_treatment) ?? Number.POSITIVE_INFINITY) <= 4).length;
  const topSite = visible[0] ? formatText(visible[0].sitename) : "--";

  document.getElementById("stats-grid").innerHTML = `
    <article class="metric-card">
      <span>Visible sites</span>
      <strong>${formatInteger(visible.length)}</strong>
      <span>${escapeHtml(state.area === "all" ? "All visible areas" : state.area)}</span>
    </article>
    <article class="metric-card">
      <span>Critical on active metric</span>
      <strong>${formatInteger(critical)}</strong>
      <span>${escapeHtml(METRICS[state.currentMetric].label)}</span>
    </article>
    <article class="metric-card">
      <span>Currently over limit</span>
      <strong>${formatInteger(overLimit)}</strong>
      <span>Female-adult lice above the current weekly threshold</span>
    </article>
    <article class="metric-card">
      <span>Treated in last 4 weeks</span>
      <strong>${formatInteger(treated)}</strong>
      <span>Top site right now: ${escapeHtml(topSite)}</span>
    </article>
  `;

  document.getElementById("visible-summary").textContent = `${visible.length} visible`;
  document.getElementById("queue-summary").textContent = visible.length
    ? `Sorted by ${METRICS[state.currentMetric].label}`
    : "No visible sites";
}

function renderSiteList() {
  const container = document.getElementById("site-list");
  const visible = state.visible.slice(0, 60);
  if (!visible.length) {
    container.innerHTML = '<div class="site-card"><div class="site-meta">No sites match the current filter set.</div></div>';
    return;
  }

  container.innerHTML = visible
    .map((site) => {
      const selectedClass = site.id === state.selectedId ? "selected" : "";
      const risk = riskBucket(metricValue(site));
      return `
        <button type="button" class="site-card ${selectedClass}" data-site-id="${escapeHtml(site.id)}">
          <div class="site-card-top">
            <div>
              <strong>${escapeHtml(formatText(site.sitename))}</strong>
              <div class="site-meta">${escapeHtml(formatText(site.productionarea))} | ${escapeHtml(formatText(site.county))}</div>
            </div>
            <strong class="metric-number">${escapeHtml(formatMetric(site))}</strong>
          </div>
          <div class="site-card-bottom">
            <span class="status-pill ${risk}">${escapeHtml(formatStatusLabel(risk))}</span>
            <span class="site-meta">Current ratio ${formatNumber(site.femaleadult_to_limit_ratio)}x</span>
            <span class="site-meta">Last treatment ${escapeHtml(formatDate(site.last_treatment_date))}</span>
          </div>
        </button>
      `;
    })
    .join("");

  container.querySelectorAll("[data-site-id]").forEach((button) => {
    button.addEventListener("click", () => selectSite(button.dataset.siteId, true));
  });
}

function detailSection(title, content, open = false) {
  return `
    <section class="detail-section-card">
      <details ${open ? "open" : ""}>
        <summary>${escapeHtml(title)}</summary>
        <div class="section-content">${content}</div>
      </details>
    </section>
  `;
}

function detailKpi(label, value) {
  return `
    <div class="detail-kpi">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `;
}

function forecastCard(site, horizon) {
  const score = site[`classifier_${horizon}w_score`];
  const count = site[`count_${horizon}w_prediction`];
  const risk = riskBucket(score, `classifier_${horizon}w_score`);
  return `
    <div class="detail-kpi">
      <span>${horizon}w breach risk</span>
      <strong>${escapeHtml(formatPercent(score))}</strong>
      <span class="mini-pill ${risk}">${escapeHtml(formatStatusLabel(risk))}</span>
      <span>Predicted count ${escapeHtml(formatNumber(count))}</span>
    </div>
  `;
}

function renderDetailPanel(site) {
  const panel = document.getElementById("detail-panel");
  if (!site) {
    panel.innerHTML = `
      <div class="detail-empty">
        <p class="eyebrow">Selected site</p>
        <h2>No site selected</h2>
        <p class="detail-copy">
          Pick a site from the priority queue, click a dot on the map, or ask the analyst for an exact match or near-miss.
        </p>
      </div>
    `;
    return;
  }

  const risk = riskBucket(metricValue(site));
  panel.innerHTML = `
    <div class="detail-shell">
      <section class="detail-hero">
        <div class="detail-topline">
          <div>
            <p class="eyebrow">Selected site</p>
            <h2>${escapeHtml(formatText(site.sitename))}</h2>
            <p class="detail-copy">${escapeHtml(formatText(site.productionarea))} | ${escapeHtml(formatText(site.county))} | Site ${escapeHtml(formatText(site.sitenumber))}</p>
          </div>
          <span class="status-pill ${risk}">${escapeHtml(formatStatusLabel(risk))}</span>
        </div>
        <div class="detail-actions">
          <span class="badge">${escapeHtml(METRICS[state.currentMetric].label)}: ${escapeHtml(formatMetric(site))}</span>
          <span class="badge">Coords ${escapeHtml(formatNumber(site.latitude, 4))}, ${escapeHtml(formatNumber(site.longitude, 4))}</span>
        </div>
        <div class="detail-kpis">
          <div class="detail-kpi">
            <span>Current female adult</span>
            <strong>${escapeHtml(formatNumber(site.femaleadult))}</strong>
          </div>
          <div class="detail-kpi">
            <span>Limit ratio</span>
            <strong>${escapeHtml(formatNumber(site.femaleadult_to_limit_ratio))}x</strong>
          </div>
          <div class="detail-kpi">
            <span>Currently over limit</span>
            <strong>${escapeHtml(formatBoolean(site.currently_over_limit))}</strong>
          </div>
        </div>
      </section>

      ${detailSection(
        "Forecast horizon cards",
        `
          <div class="forecast-grid">
            ${forecastCard(site, 1)}
            ${forecastCard(site, 2)}
            ${forecastCard(site, 12)}
          </div>
        `,
        true,
      )}

      ${detailSection(
        "Current status",
        `
          <div class="detail-grid">
            ${detailKpi("Latest observation", formatDate(site.latest_observation_date))}
            ${detailKpi("Sea temperature", formatNumber(site.seatemperature))}
            ${detailKpi("Mobile lice", formatNumber(site.mobilelice))}
            ${detailKpi("Persistent lice", formatNumber(site.persistentlice))}
            ${detailKpi("Counted lice this week", formatBoolean(site.havecountedlice))}
            ${detailKpi("Likely no fish", formatBoolean(site.likelynofish))}
          </div>
        `,
        true,
      )}

      ${detailSection(
        "Treatment context",
        `
          <div class="detail-grid">
            ${detailKpi("Last treatment date", formatDate(site.last_treatment_date))}
            ${detailKpi("Last treatment action", formatText(site.last_treatment_action))}
            ${detailKpi("Active ingredient", formatText(site.last_treatment_activeingredient))}
            ${detailKpi("Weeks since any treatment", formatNumber(site.weeks_since_any_treatment, 0))}
            ${detailKpi("This-week treatment count", formatNumber(site.treatment_count, 0))}
            ${detailKpi("Cleaner fish treatments", formatNumber(site.cleanerfish_treatment_count, 0))}
          </div>
        `,
        true,
      )}

      ${detailSection(
        "Pressure context",
        `
          <div class="detail-grid">
            ${detailKpi("Area breach rate lag1", formatPercent(site.pa_breach_rate_lag1))}
            ${detailKpi("Area treatment rate lag1", formatPercent(site.pa_treatment_rate_lag1))}
            ${detailKpi("Neighbor breach rate lag1", formatPercent(site.neighbor_breach_this_week_lag1))}
            ${detailKpi("Neighbor limit ratio lag1", `${formatNumber(site.neighbor_femaleadult_to_limit_ratio_lag1)}x`)}
            ${detailKpi("Neighbor sites in radius", formatNumber(site.neighbor_site_count, 0))}
            ${detailKpi("Priority horizon", formatText(site.priority_horizon))}
          </div>
        `,
        true,
      )}
    </div>
  `;
}

function syncLayoutMetrics() {
  const topbar = document.querySelector(".topbar");
  if (!topbar) {
    return;
  }

  const root = document.documentElement;
  const chromeTop = Math.ceil(topbar.getBoundingClientRect().bottom + 14);
  root.style.setProperty("--chrome-top", `${chromeTop}px`);
}

function buildGeojson() {
  return {
    type: "FeatureCollection",
    features: state.visible.map((site) => {
      const value = metricValue(site);
      const bucket = riskBucket(value);
      return {
        type: "Feature",
        properties: {
          id: site.id,
          sitename: formatText(site.sitename),
          productionarea: formatText(site.productionarea),
          metricDisplay: formatMetric(site),
          metricLabel: METRICS[state.currentMetric].label,
          markerColor: bucketColor(bucket),
          markerRadius: markerRadius(value),
          selected: site.id === state.selectedId,
        },
        geometry: {
          type: "Point",
          coordinates: [site.longitude, site.latitude],
        },
      };
    }),
  };
}

function updateMapSource() {
  const source = map.getSource("sites");
  if (!source) {
    return;
  }
  source.setData(buildGeojson());
}

function fitVisibleSites() {
  if (!state.visible.length) {
    return;
  }
  const bounds = new maplibregl.LngLatBounds();
  state.visible.forEach((site) => bounds.extend([site.longitude, site.latitude]));
  map.fitBounds(bounds, { padding: 70, duration: 700, maxZoom: 9.25 });
}

function openDetailPanel() {
  document.getElementById("detail-panel-shell").classList.add("open");
}

function setChatOpen(nextOpen) {
  state.chatOpen = nextOpen;
  const drawer = document.getElementById("chat-dock");
  const backdrop = document.getElementById("chat-backdrop");
  const openButton = document.getElementById("open-chat-button");
  const toggleButton = document.getElementById("toggle-chat-button");

  drawer.classList.toggle("open", nextOpen);
  backdrop.classList.toggle("visible", nextOpen);
  openButton.classList.toggle("visible", !nextOpen);
  toggleButton.textContent = nextOpen ? "Hide panel" : "Open panel";

  if (nextOpen) {
    document.getElementById("chat-input").focus();
  }
}

function syncChatStatus(text) {
  document.getElementById("chat-status").textContent = text;
}

function selectSite(siteId, flyToSite = false) {
  const match = state.visible.find((site) => site.id === String(siteId)) || null;
  state.selectedId = match ? match.id : null;
  renderSiteList();
  renderDetailPanel(match || state.visible[0] || null);
  updateMapSource();
  openDetailPanel();
  if (match && flyToSite) {
    map.flyTo({ center: [match.longitude, match.latitude], zoom: 8.5, duration: 750 });
  }
}

function refreshView(options = {}) {
  state.visible = sortSites(filterSites());
  if (!state.visible.find((site) => site.id === state.selectedId)) {
    state.selectedId = state.visible[0] ? state.visible[0].id : null;
  }

  renderMetricSwitcher();
  syncLayoutMetrics();
  renderRiskFilter();
  buildLegend();
  renderStats();
  renderSiteList();
  renderDetailPanel(state.visible.find((site) => site.id === state.selectedId) || null);
  updateMapSource();

  if (options.fitBounds) {
    fitVisibleSites();
  }
}

function closePanelsOnResize() {
  syncLayoutMetrics();
  if (window.innerWidth > 900) {
    document.getElementById("filters-panel").classList.remove("open");
  }
  if (window.innerWidth > 1180) {
    document.getElementById("detail-panel-shell").classList.remove("open");
  }
}

function appendChatMessage(role, html) {
  const transcript = document.getElementById("chat-transcript");
  const message = document.createElement("div");
  message.className = `chat-message ${role}`;
  message.innerHTML = html;
  transcript.appendChild(message);
  transcript.scrollTop = transcript.scrollHeight;
}

function renderSuggestionChips() {
  document.getElementById("chat-suggestions").innerHTML = CHAT_SUGGESTIONS.map(
    (suggestion) => `<button type="button" class="suggestion-chip">${escapeHtml(suggestion)}</button>`,
  ).join("");

  document.querySelectorAll(".suggestion-chip").forEach((button) => {
    button.addEventListener("click", () => {
      document.getElementById("chat-input").value = button.textContent;
      setChatOpen(true);
    });
  });
}

function renderChatResult(payload) {
  const siteCards = (payload.sites || [])
    .map(
      (site) => `
        <article class="result-card">
          <div class="result-card-top">
            <div>
              <strong>${escapeHtml(formatText(site.sitename))}</strong>
              <div class="site-meta">${escapeHtml(formatText(site.productionarea))} | ${escapeHtml(formatText(site.county))}</div>
            </div>
            <strong>${escapeHtml(formatText(site.metric_display))}</strong>
          </div>
          <div class="site-card-bottom">
            <span class="site-meta">Coords ${escapeHtml(formatText(site.coordinates_text))}</span>
            <span class="site-meta">Current ratio ${formatNumber(site.femaleadult_to_limit_ratio)}x</span>
          </div>
          <div class="result-actions">
            <button type="button" class="ghost-button compact" data-chat-site="${escapeHtml(site.site_id)}">Take me there</button>
          </div>
        </article>
      `,
    )
    .join("");

  appendChatMessage(
    "assistant",
    `
      <div class="message-row">
        <strong>Atlas assistant</strong>
        <span class="badge">${payload.used_llm ? "Gemini" : "Fallback"}</span>
      </div>
      <div class="message-copy">${renderMarkdown(payload.answer)}</div>
      ${payload.proxy_note ? `<div class="popup-line">${escapeHtml(payload.proxy_note)}</div>` : ""}
      ${payload.filters_applied?.length ? `<div class="popup-line">Filters: ${payload.filters_applied.map((value) => escapeHtml(value)).join(", ")}</div>` : ""}
      ${siteCards ? `<div class="result-list">${siteCards}</div>` : ""}
    `,
  );

  const llm = payload.llm || {};
  if (payload.used_llm) {
    syncChatStatus(`Gemini live on Vertex: ${llm.model || "configured model"}.`);
  } else if (llm.last_error) {
    syncChatStatus(`Fallback mode: ${llm.last_error}`);
  } else {
    syncChatStatus("Fallback mode: deterministic site ranking.");
  }

  document.querySelectorAll("[data-chat-site]").forEach((button) => {
    button.addEventListener("click", () => selectSite(button.dataset.chatSite, true));
  });
}

function appendLoadingMessage() {
  appendChatMessage(
    "assistant",
    `
      <div class="message-row">
        <strong>Atlas assistant</strong>
        <span class="badge">Thinking</span>
      </div>
      <div class="message-copy"><p>Looking through the visible sites and preparing the answer.</p></div>
    `,
  );
}

function removeLastLoadingMessage() {
  const transcript = document.getElementById("chat-transcript");
  const messages = transcript.querySelectorAll(".chat-message");
  const last = messages[messages.length - 1];
  if (!last) {
    return;
  }
  if (last.textContent.includes("Looking through the visible sites")) {
    last.remove();
  }
}

async function submitChat(message) {
  const submitButton = document.getElementById("chat-submit-button");
  setChatOpen(true);
  submitButton.disabled = true;
  appendChatMessage("user", `<div class="message-copy">${renderMarkdown(message)}</div>`);
  appendLoadingMessage();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        selected_site_id: state.selectedId,
        visible_site_ids: state.visible.map((site) => site.id),
      }),
    });
    if (!response.ok) {
      throw new Error(`Chat request failed with ${response.status}`);
    }
    const payload = await response.json();
    removeLastLoadingMessage();
    renderChatResult(payload);
  } catch (error) {
    removeLastLoadingMessage();
    syncChatStatus("Chat request failed.");
    appendChatMessage(
      "assistant",
      `<div class="message-copy"><p>Chat request failed: ${escapeHtml(error.message)}</p></div>`,
    );
  } finally {
    submitButton.disabled = false;
  }
}

function wireControls() {
  document.getElementById("area-filter").addEventListener("change", (event) => {
    state.area = event.target.value;
    refreshView({ fitBounds: true });
  });

  document.getElementById("county-filter").addEventListener("change", (event) => {
    state.county = event.target.value;
    refreshView({ fitBounds: true });
  });

  document.getElementById("search-input").addEventListener("input", (event) => {
    state.query = event.target.value;
    refreshView();
  });

  document.getElementById("over-limit-toggle").addEventListener("change", (event) => {
    state.overLimitOnly = event.target.checked;
    refreshView({ fitBounds: true });
  });

  document.getElementById("recent-treatment-toggle").addEventListener("change", (event) => {
    state.recentTreatmentOnly = event.target.checked;
    refreshView({ fitBounds: true });
  });

  document.getElementById("counted-only-toggle").addEventListener("change", (event) => {
    state.countedOnly = event.target.checked;
    refreshView({ fitBounds: true });
  });

  document.getElementById("fit-visible-button").addEventListener("click", () => {
    fitVisibleSites();
  });

  document.getElementById("reset-filters-button").addEventListener("click", () => {
    state.currentMetric = "near_term_risk";
    state.riskFilter = "all";
    state.area = "all";
    state.county = "all";
    state.query = "";
    state.overLimitOnly = false;
    state.recentTreatmentOnly = false;
    state.countedOnly = false;
    document.getElementById("area-filter").value = "all";
    document.getElementById("county-filter").value = "all";
    document.getElementById("search-input").value = "";
    document.getElementById("over-limit-toggle").checked = false;
    document.getElementById("recent-treatment-toggle").checked = false;
    document.getElementById("counted-only-toggle").checked = false;
    refreshView({ fitBounds: true });
  });

  document.getElementById("mobile-filters-button").addEventListener("click", () => {
    document.getElementById("filters-panel").classList.toggle("open");
  });

  document.getElementById("mobile-details-button").addEventListener("click", () => {
    document.getElementById("detail-panel-shell").classList.toggle("open");
  });

  document.getElementById("toggle-chat-button").addEventListener("click", () => {
    setChatOpen(false);
  });

  document.getElementById("open-chat-button").addEventListener("click", () => {
    setChatOpen(true);
  });

  document.getElementById("chat-backdrop").addEventListener("click", () => {
    setChatOpen(false);
  });

  document.getElementById("chat-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) {
      return;
    }
    input.value = "";
    await submitChat(message);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.chatOpen) {
      setChatOpen(false);
    }
  });

  window.addEventListener("resize", closePanelsOnResize);
}

function addMapLayers() {
  map.addSource("sites", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });

  map.addLayer({
    id: "site-shadow",
    type: "circle",
    source: "sites",
    paint: {
      "circle-radius": ["+", ["get", "markerRadius"], 3],
      "circle-color": "rgba(0, 0, 0, 0.3)",
      "circle-blur": 0.25,
    },
  });

  map.addLayer({
    id: "site-dots",
    type: "circle",
    source: "sites",
    filter: ["!", ["get", "selected"]],
    paint: {
      "circle-radius": ["get", "markerRadius"],
      "circle-color": ["get", "markerColor"],
      "circle-opacity": 0.92,
      "circle-stroke-width": 1.5,
      "circle-stroke-color": "rgba(255, 255, 255, 0.32)",
    },
  });

  map.addLayer({
    id: "site-selected",
    type: "circle",
    source: "sites",
    filter: ["get", "selected"],
    paint: {
      "circle-radius": ["+", ["get", "markerRadius"], 5],
      "circle-color": ["get", "markerColor"],
      "circle-opacity": 1,
      "circle-stroke-width": 3,
      "circle-stroke-color": "#ffffff",
    },
  });

  map.on("mouseenter", "site-dots", (event) => {
    map.getCanvas().style.cursor = "pointer";
    const feature = event.features[0];
    hoverPopup
      .setLngLat(feature.geometry.coordinates)
      .setHTML(`
        <strong>${escapeHtml(feature.properties.sitename)}</strong>
        <div class="popup-line">${escapeHtml(feature.properties.productionarea)}</div>
        <div class="popup-line">${escapeHtml(feature.properties.metricLabel)}: ${escapeHtml(feature.properties.metricDisplay)}</div>
      `)
      .addTo(map);
  });

  map.on("mouseleave", "site-dots", () => {
    map.getCanvas().style.cursor = "";
    hoverPopup.remove();
  });

  ["site-dots", "site-selected"].forEach((layerId) => {
    map.on("click", layerId, (event) => {
      const feature = event.features[0];
      selectSite(feature.properties.id, true);
    });
  });
}

map.on("load", async () => {
  addMapLayers();
  try {
    const dataset = await fetchSiteDataset();
    state.datasetMeta = dataset.metadata || {};
    state.features = (dataset.features || []).map(normalizeFeature);
    populateFilters();
    wireControls();
    renderSuggestionChips();
    renderCaseDate();
    syncChatStatus("Gemini on Vertex when available, with rule-based reasoning as a fallback.");
    setChatOpen(false);
    refreshView({ fitBounds: true });
    syncLayoutMetrics();
  } catch (error) {
    document.getElementById("detail-panel").innerHTML = `
      <div class="detail-empty">
        <p class="eyebrow">Data error</p>
        <h2>Map data could not be loaded</h2>
        <p class="detail-copy">${escapeHtml(error.message)}</p>
      </div>
    `;
    console.error(error);
  }
});