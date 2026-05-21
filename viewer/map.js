const METRICS = {
  classifier_12w_score: {
    label: "12w breach risk",
    kind: "probability",
    thresholds: [0.4, 0.6, 0.8],
  },
  classifier_2w_score: {
    label: "2w breach risk",
    kind: "probability",
    thresholds: [0.35, 0.55, 0.75],
  },
  classifier_1w_score: {
    label: "1w breach risk",
    kind: "probability",
    thresholds: [0.35, 0.55, 0.8],
  },
  femaleadult_to_limit_ratio: {
    label: "Current limit ratio",
    kind: "ratio",
    thresholds: [0.5, 1.0, 1.5],
  },
};

const state = {
  currentMetric: "classifier_12w_score",
  area: "all",
  query: "",
  features: [],
  visible: [],
  selectedId: null,
};

const map = new maplibregl.Map({
  container: "map",
  style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  center: [14.2, 65.1],
  zoom: 4.2,
  pitch: 0,
  bearing: 0,
});

map.addControl(new maplibregl.NavigationControl(), "top-right");

const hoverPopup = new maplibregl.Popup({
  closeButton: false,
  closeOnClick: false,
  offset: 14,
});

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatNumber(value, digits = 2) {
  const number = toNumber(value);
  if (number === null) {
    return "--";
  }
  return number.toFixed(digits);
}

function formatInteger(value) {
  const number = toNumber(value);
  if (number === null) {
    return "--";
  }
  return Math.round(number).toString();
}

function formatPercent(value) {
  const number = toNumber(value);
  if (number === null) {
    return "--";
  }
  return `${Math.round(number * 100)}%`;
}

function formatMetric(site, metricKey = state.currentMetric) {
  const metric = METRICS[metricKey];
  const value = toNumber(site[metricKey]);
  if (value === null) {
    return "--";
  }
  if (metric.kind === "probability") {
    return formatPercent(value);
  }
  return `${formatNumber(value)}x`;
}

function riskBucket(value, metricKey = state.currentMetric) {
  const metric = METRICS[metricKey];
  const numericValue = toNumber(value);
  if (numericValue === null) {
    return "stable";
  }
  if (numericValue >= metric.thresholds[2]) {
    return "critical";
  }
  if (numericValue >= metric.thresholds[1]) {
    return "high";
  }
  if (numericValue >= metric.thresholds[0]) {
    return "watch";
  }
  return "stable";
}

function bucketColor(bucket) {
  if (bucket === "critical") {
    return "#ff3b30";
  }
  if (bucket === "high") {
    return "#ffb347";
  }
  if (bucket === "watch") {
    return "#39b9b2";
  }
  return "#7b8c94";
}

function markerRadius(value, metricKey = state.currentMetric) {
  const numericValue = toNumber(value);
  if (numericValue === null) {
    return 6;
  }
  const metric = METRICS[metricKey];
  if (metric.kind === "probability") {
    return 7 + Math.min(12, numericValue * 14);
  }
  return 7 + Math.min(12, numericValue * 4.5);
}

function formatBoolean(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  return value ? "Yes" : "No";
}

function formatText(value) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  return String(value);
}

function metricValue(site) {
  return toNumber(site[state.currentMetric]);
}

function normalizeFeature(feature) {
  const properties = feature.properties || {};
  const geometry = feature.geometry || { coordinates: [null, null] };
  return {
    ...properties,
    id: String(properties.sitenumber),
    longitude: geometry.coordinates[0],
    latitude: geometry.coordinates[1],
  };
}

function buildLegend() {
  const metric = METRICS[state.currentMetric];
  const labels = [
    { label: `Below ${formatLegendValue(metric.thresholds[0], metric.kind)}`, bucket: "stable" },
    { label: `${formatLegendValue(metric.thresholds[0], metric.kind)} to ${formatLegendValue(metric.thresholds[1], metric.kind)}`, bucket: "watch" },
    { label: `${formatLegendValue(metric.thresholds[1], metric.kind)} to ${formatLegendValue(metric.thresholds[2], metric.kind)}`, bucket: "high" },
    { label: `Above ${formatLegendValue(metric.thresholds[2], metric.kind)}`, bucket: "critical" },
  ];

  document.getElementById("legend").innerHTML = labels
    .map(
      (item) => `
        <div class="legend-row">
          <span>${item.label}</span>
          <span class="legend-swatch" style="background:${bucketColor(item.bucket)}"></span>
        </div>
      `,
    )
    .join("");
}

function formatLegendValue(value, kind) {
  if (kind === "probability") {
    return formatPercent(value);
  }
  return `${formatNumber(value)}x`;
}

function filterFeatures() {
  const query = state.query.trim().toLowerCase();
  return state.features.filter((site) => {
    const matchesArea = state.area === "all" || site.productionarea === state.area;
    const matchesQuery =
      query === "" ||
      formatText(site.sitename).toLowerCase().includes(query) ||
      formatText(site.sitenumber).toLowerCase().includes(query);
    return matchesArea && matchesQuery;
  });
}

function sortSites(sites) {
  return [...sites].sort((left, right) => {
    const metricGap = (metricValue(right) ?? -1) - (metricValue(left) ?? -1);
    if (metricGap !== 0) {
      return metricGap;
    }
    const leftRatio = toNumber(left.femaleadult_to_limit_ratio) ?? -1;
    const rightRatio = toNumber(right.femaleadult_to_limit_ratio) ?? -1;
    if (rightRatio !== leftRatio) {
      return rightRatio - leftRatio;
    }
    return formatText(left.sitename).localeCompare(formatText(right.sitename));
  });
}

function renderStats() {
  const visible = state.visible;
  const metricLabel = METRICS[state.currentMetric].label;
  const criticalCount = visible.filter(
    (site) => riskBucket(metricValue(site)) === "critical",
  ).length;
  const overLimitCount = visible.filter((site) => Boolean(site.currently_over_limit)).length;
  const recentTreatmentCount = visible.filter((site) => {
    const weeks = toNumber(site.weeks_since_any_treatment);
    return weeks !== null && weeks <= 4;
  }).length;
  const latestDate = visible.reduce((latest, site) => {
    const candidate = site.latest_observation_date;
    if (!candidate) {
      return latest;
    }
    return latest === null || candidate > latest ? candidate : latest;
  }, null);

  document.getElementById("stats-grid").innerHTML = `
    <div class="stat-card">
      <span>Sites on map</span>
      <strong>${formatInteger(visible.length)}</strong>
    </div>
    <div class="stat-card">
      <span>${metricLabel} critical</span>
      <strong>${formatInteger(criticalCount)}</strong>
    </div>
    <div class="stat-card">
      <span>Currently over limit</span>
      <strong>${formatInteger(overLimitCount)}</strong>
    </div>
    <div class="stat-card">
      <span>Latest observation</span>
      <strong>${latestDate || "--"}</strong>
      <span>${formatInteger(recentTreatmentCount)} sites treated in the last 4 weeks</span>
    </div>
  `;
}

function renderSiteList() {
  const siteList = document.getElementById("site-list");
  const visible = state.visible.slice(0, 80);
  document.getElementById("site-count-pill").textContent = `${state.visible.length} visible`;

  if (visible.length === 0) {
    siteList.innerHTML = '<div class="site-item"><div class="site-meta">No sites match the current filters.</div></div>';
    return;
  }

  siteList.innerHTML = visible
    .map((site) => {
      const selectedClass = site.id === state.selectedId ? "selected" : "";
      return `
        <button class="site-item ${selectedClass}" data-site-id="${site.id}">
          <div class="site-topline">
            <div>
              <strong>${formatText(site.sitename)}</strong>
              <div class="site-area">${formatText(site.productionarea)}</div>
            </div>
            <strong class="site-metric">${formatMetric(site)}</strong>
          </div>
          <div class="site-bottomline">
            <span class="pill risk-${riskBucket(metricValue(site))}">${riskBucket(metricValue(site))}</span>
            <span class="site-meta">Current female adult ${formatNumber(site.femaleadult)}</span>
            <span class="site-meta">Limit ratio ${formatNumber(site.femaleadult_to_limit_ratio)}x</span>
          </div>
        </button>
      `;
    })
    .join("");

  siteList.querySelectorAll("[data-site-id]").forEach((button) => {
    button.addEventListener("click", () => selectSite(button.dataset.siteId, true));
  });
}

function buildForecastCard(site, horizon) {
  const score = site[`classifier_${horizon}w_score`];
  const prediction = site[`classifier_${horizon}w_prediction`];
  const actual = site[`classifier_${horizon}w_actual`];
  const countPrediction = site[`count_${horizon}w_prediction`];
  const countActual = site[`count_${horizon}w_actual`];
  const date = site[`classifier_${horizon}w_date`] || site[`count_${horizon}w_date`];
  return `
    <div class="forecast-card">
      <span>${horizon}w forecast</span>
      <strong>${formatPercent(score)}</strong>
      <div class="popup-line">Date ${formatText(date)}</div>
      <div class="popup-line">Predicted breach ${formatBoolean(prediction)}</div>
      <div class="popup-line">Actual breach ${formatBoolean(actual)}</div>
      <div class="popup-line">Predicted count ${formatNumber(countPrediction)}</div>
      <div class="popup-line">Actual count ${formatNumber(countActual)}</div>
    </div>
  `;
}

function renderDetails(site) {
  const detailPanel = document.getElementById("detail-panel");
  if (!site) {
    detailPanel.innerHTML = '<div class="detail-block"><div class="detail-muted">Select a site to inspect its latest status, treatment context, and forecast track record.</div></div>';
    return;
  }

  detailPanel.innerHTML = `
    <div class="detail-block">
      <div class="section-header">
        <div>
          <h3>${formatText(site.sitename)}</h3>
          <p class="detail-muted">${formatText(site.productionarea)} | Site ${formatText(site.sitenumber)}</p>
        </div>
        <span class="pill risk-${riskBucket(metricValue(site))}">${riskBucket(metricValue(site))}</span>
      </div>
      <div class="detail-chip-row">
        <div class="detail-chip"><span>Observation date</span><strong>${formatText(site.latest_observation_date)}</strong></div>
        <div class="detail-chip"><span>Current metric</span><strong>${formatMetric(site)}</strong></div>
        <div class="detail-chip"><span>Priority horizon</span><strong>${formatText(site.priority_horizon)}</strong></div>
      </div>
    </div>

    <div class="detail-block">
      <h3>Current status</h3>
      <div class="detail-grid">
        <div class="detail-chip"><span>Female adult</span><strong>${formatNumber(site.femaleadult)}</strong></div>
        <div class="detail-chip"><span>Limit ratio</span><strong>${formatNumber(site.femaleadult_to_limit_ratio)}x</strong></div>
        <div class="detail-chip"><span>Mobile lice</span><strong>${formatNumber(site.mobilelice)}</strong></div>
        <div class="detail-chip"><span>Persistent lice</span><strong>${formatNumber(site.persistentlice)}</strong></div>
        <div class="detail-chip"><span>Currently over limit</span><strong>${formatBoolean(site.currently_over_limit)}</strong></div>
        <div class="detail-chip"><span>Sea temperature</span><strong>${formatNumber(site.seatemperature)}</strong></div>
      </div>
    </div>

    <div class="detail-block">
      <h3>Treatment context</h3>
      <div class="detail-grid">
        <div class="detail-chip"><span>Last treatment date</span><strong>${formatText(site.last_treatment_date)}</strong></div>
        <div class="detail-chip"><span>Last treatment action</span><strong>${formatText(site.last_treatment_action)}</strong></div>
        <div class="detail-chip"><span>Active ingredient</span><strong>${formatText(site.last_treatment_activeingredient)}</strong></div>
        <div class="detail-chip"><span>Weeks since any treatment</span><strong>${formatNumber(site.weeks_since_any_treatment, 0)}</strong></div>
        <div class="detail-chip"><span>This-week treatment count</span><strong>${formatNumber(site.treatment_count, 0)}</strong></div>
        <div class="detail-chip"><span>Cleaner fish treatments</span><strong>${formatNumber(site.cleanerfish_treatment_count, 0)}</strong></div>
      </div>
    </div>

    <div class="detail-block">
      <h3>Pressure context</h3>
      <div class="detail-grid">
        <div class="detail-chip"><span>Area breach rate lag1</span><strong>${formatPercent(site.pa_breach_rate_lag1)}</strong></div>
        <div class="detail-chip"><span>Area treatment rate lag1</span><strong>${formatPercent(site.pa_treatment_rate_lag1)}</strong></div>
        <div class="detail-chip"><span>Neighbor breach rate lag1</span><strong>${formatPercent(site.neighbor_breach_this_week_lag1)}</strong></div>
        <div class="detail-chip"><span>Neighbor limit ratio lag1</span><strong>${formatNumber(site.neighbor_femaleadult_to_limit_ratio_lag1)}x</strong></div>
        <div class="detail-chip"><span>Neighbor sites in radius</span><strong>${formatNumber(site.neighbor_site_count, 0)}</strong></div>
        <div class="detail-chip"><span>Likely no fish</span><strong>${formatBoolean(site.likelynofish)}</strong></div>
      </div>
    </div>

    <div class="detail-block">
      <h3>Validated forecast track record</h3>
      <div class="forecast-grid">
        ${buildForecastCard(site, 1)}
        ${buildForecastCard(site, 2)}
        ${buildForecastCard(site, 12)}
      </div>
    </div>
  `;
}

function updateMapSource() {
  const source = map.getSource("sites");
  if (!source) {
    return;
  }
  const geojson = {
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
          metricValue: value,
          metricLabel: METRICS[state.currentMetric].label,
          metricDisplay: formatMetric(site),
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
  source.setData(geojson);
}

function fitToVisibleSites() {
  if (state.visible.length === 0) {
    return;
  }
  const bounds = new maplibregl.LngLatBounds();
  state.visible.forEach((site) => bounds.extend([site.longitude, site.latitude]));
  map.fitBounds(bounds, { padding: 60, maxZoom: 9, duration: 700 });
}

function selectSite(siteId, flyToSite) {
  const next = state.visible.find((site) => site.id === String(siteId)) || null;
  state.selectedId = next ? next.id : null;
  renderSiteList();
  renderDetails(next || state.visible[0] || null);
  updateMapSource();
  if (next && flyToSite) {
    map.flyTo({ center: [next.longitude, next.latitude], zoom: 8.8, duration: 800 });
  }
}

function refreshView(options = {}) {
  state.visible = sortSites(filterFeatures());
  if (!state.visible.find((site) => site.id === state.selectedId)) {
    state.selectedId = state.visible[0] ? state.visible[0].id : null;
  }
  renderStats();
  renderSiteList();
  renderDetails(state.visible.find((site) => site.id === state.selectedId) || null);
  document.getElementById("metric-pill").textContent = METRICS[state.currentMetric].label;
  document.getElementById("map-title").textContent = METRICS[state.currentMetric].label;
  document.getElementById("top-site-label").textContent = state.visible[0]
    ? formatText(state.visible[0].sitename)
    : "No sites visible";
  buildLegend();
  updateMapSource();
  if (options.fitBounds) {
    fitToVisibleSites();
  }
}

function populateAreaFilter() {
  const select = document.getElementById("area-filter");
  const areas = [...new Set(state.features.map((site) => site.productionarea).filter(Boolean))].sort();
  select.innerHTML = ['<option value="all">All production areas</option>']
    .concat(areas.map((area) => `<option value="${area}">${area}</option>`))
    .join("");
}

function wireControls() {
  document.querySelectorAll(".metric-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".metric-button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.currentMetric = button.dataset.metric;
      refreshView();
    });
  });

  document.getElementById("area-filter").addEventListener("change", (event) => {
    state.area = event.target.value;
    refreshView({ fitBounds: true });
  });

  document.getElementById("search-input").addEventListener("input", (event) => {
    state.query = event.target.value;
    refreshView();
  });
}

function addMapLayers() {
  map.addSource("sites", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });

  map.addLayer({
    id: "site-dots-shadow",
    type: "circle",
    source: "sites",
    paint: {
      "circle-radius": ["+", ["get", "markerRadius"], 2],
      "circle-color": "rgba(0, 0, 0, 0.28)",
      "circle-blur": 0.2,
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
    id: "site-dots-selected",
    type: "circle",
    source: "sites",
    filter: ["get", "selected"],
    paint: {
      "circle-radius": ["+", ["get", "markerRadius"], 4],
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
        <strong>${feature.properties.sitename}</strong>
        <div class="popup-line">${feature.properties.productionarea}</div>
        <div class="popup-line">${feature.properties.metricLabel}: ${feature.properties.metricDisplay}</div>
      `)
      .addTo(map);
  });

  map.on("mouseleave", "site-dots", () => {
    map.getCanvas().style.cursor = "";
    hoverPopup.remove();
  });

  ["site-dots", "site-dots-selected"].forEach((layerId) => {
    map.on("click", layerId, (event) => {
      const feature = event.features[0];
      selectSite(feature.properties.id, true);
    });
  });
}

async function loadDataset() {
  const response = await fetch("../results/site_map.geojson");
  if (!response.ok) {
    throw new Error(`Failed to load map data: ${response.status}`);
  }
  const data = await response.json();
  state.features = data.features.map(normalizeFeature);
}

map.on("load", async () => {
  addMapLayers();
  try {
    await loadDataset();
    populateAreaFilter();
    wireControls();
    refreshView({ fitBounds: true });
  } catch (error) {
    document.getElementById("detail-panel").innerHTML = `<div class="detail-block"><div class="detail-muted">${error.message}</div></div>`;
    console.error(error);
  }
});