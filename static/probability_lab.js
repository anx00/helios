(() => {
    const page = document.getElementById("plPage");
    if (!page) {
        return;
    }

    const state = {
        stationId: null,
        targetDay: 1,
        payload: null,
        chart: null,
        requestId: 0,
    };

    const els = {
        toggle: document.getElementById("plToggle"),
        contextChips: document.getElementById("plContextChips"),
        stationTitle: document.getElementById("plStationTitle"),
        stationMeta: document.getElementById("plStationMeta"),
        stationContext: document.getElementById("plStationContext"),
        actionBadge: document.getElementById("plActionBadge"),
        actionTitle: document.getElementById("plActionTitle"),
        actionCopy: document.getElementById("plActionCopy"),
        actionMeta: document.getElementById("plActionMeta"),
        likelyCard: document.getElementById("plLikelyCard"),
        marketCard: document.getElementById("plMarketCard"),
        realityCard: document.getElementById("plRealityCard"),
        whyList: document.getElementById("plWhyList"),
        advancedDetails: document.getElementById("plAdvancedDetails"),
        advancedSummary: document.getElementById("plAdvancedSummary"),
        historyMeta: document.getElementById("plHistoryMeta"),
        historyChart: document.getElementById("plHistoryChart"),
        sourceList: document.getElementById("plSourceList"),
        ladder: document.getElementById("plLadder"),
        realityDetails: document.getElementById("plRealityDetails"),
        calibration: document.getElementById("plCalibration"),
        pws: document.getElementById("plPws"),
    };

    const sourceColors = {
        WUNDERGROUND: "#f59e0b",
        OPEN_METEO: "#38bdf8",
        NBM: "#34d399",
        LAMP: "#f472b6",
    };

    const realityLabels = {
        official_temp_market: "Official temperature",
        official_temp_c: "Official temp (C)",
        official_temp_f: "Official temp (F)",
        official_obs_time_utc: "Official observation",
        cumulative_max_market: "Running day max",
        cumulative_max_f: "Running day max (F)",
        current_reality_bracket: "Current reality bracket",
        resolved_winning_bracket: "Resolved winning bracket",
        calibration_state: "Calibration state",
        calibration_samples: "Calibration samples",
        calibration_mae_market: "Calibration MAE",
        calibration_bias_market: "Calibration bias",
    };

    const directReasonMap = {
        "No trade candidate": "No setup clears the minimum bar for a trade right now.",
        "No actionable edge": "The current setup does not clear the minimum edge needed to act.",
        "Late upside chase after the expected heat peak.": "This trade would need a late upside push after the expected peak.",
        "Next official print is leaning down, so upside chase is blocked.": "The next official print is leaning cooler, so the upside setup is blocked.",
        "This bucket sits too far from the forecast winner for live trading.": "This bracket is too far away from the model's main scenario.",
        "The forecast winner already offers a better or similar price.": "There is a better-priced way to express the same view.",
        "Probability is too small for automatic trading.": "The odds are too low to justify a trade.",
        "Price is too close to max payout for the remaining edge.": "The market price is already too expensive for the remaining edge.",
        "Edge is too small after the policy thresholds.": "The edge is too small after guardrails and pricing thresholds.",
        "Missing market quotes for this side.": "There is not enough live market pricing on this side.",
        "Recent resolved days are still warming up.": "Recent resolved days are still warming up, so calibration is not fully trusted yet.",
        "No resolved-day calibration yet.": "There is not enough resolved history yet to calibrate this horizon.",
    };

    function esc(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function destroyChart() {
        if (state.chart) {
            state.chart.destroy();
            state.chart = null;
        }
    }

    function currentStationId() {
        const selected = window.HeliosStationState?.get?.();
        return String(selected || page.dataset.initialStation || "KLGA").toUpperCase();
    }

    function currentStationOption(stationId = state.stationId || currentStationId()) {
        return window.HeliosStationState?.getOption?.(stationId) || null;
    }

    function stationTimeZone() {
        return currentStationOption()?.timezone || undefined;
    }

    function fetchJSON(url) {
        return fetch(url).then(async (response) => {
            const raw = await response.text();
            let data = null;
            try {
                data = raw ? JSON.parse(raw) : {};
            } catch (_) {
                data = null;
            }
            if (!response.ok) {
                throw new Error((data && data.error) || raw || `HTTP ${response.status}`);
            }
            if (data && data.error) {
                throw new Error(data.error);
            }
            return data;
        });
    }

    function formatNumber(value, digits = 1) {
        return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "--";
    }

    function formatPercent(value, digits = 0) {
        return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "--";
    }

    function formatMoney(value) {
        return typeof value === "number" && Number.isFinite(value) ? `$${value.toFixed(2)}` : "--";
    }

    function formatDate(value) {
        if (!value) {
            return "--";
        }
        const date = new Date(`${value}T00:00:00`);
        if (Number.isNaN(date.getTime())) {
            return String(value);
        }
        return date.toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
            timeZone: stationTimeZone(),
        });
    }

    function formatStamp(value) {
        if (!value) {
            return "--";
        }
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
            return String(value);
        }
        return date.toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            timeZone: stationTimeZone(),
        });
    }

    function horizonLabel(targetDay = state.targetDay) {
        if (Number(targetDay) === 0) {
            return "Same day";
        }
        if (Number(targetDay) === 2) {
            return "Day +2";
        }
        return "Tomorrow";
    }

    function bracketLabel(label, fallback = "No clear bracket yet") {
        const value = String(label || "").trim();
        return value || fallback;
    }

    function recommendationText(code) {
        const normalized = String(code || "BLOCK").toUpperCase();
        if (normalized === "BUY_YES") {
            return "Buy YES";
        }
        if (normalized === "BUY_NO") {
            return "Buy NO";
        }
        if (normalized.startsWith("LEAN")) {
            return "Lean";
        }
        if (normalized.startsWith("WATCH")) {
            return "Watch";
        }
        return "No trade";
    }

    function sourceName(value) {
        const normalized = String(value || "").toUpperCase();
        if (normalized === "WUNDERGROUND") {
            return "Wunderground";
        }
        if (normalized === "OPEN_METEO") {
            return "Open-Meteo";
        }
        return normalized || "--";
    }

    function sourceStatusText(value) {
        const normalized = String(value || "").toLowerCase();
        if (normalized === "ok") {
            return "OK";
        }
        if (normalized === "partial") {
            return "Partial";
        }
        if (normalized === "error") {
            return "Error";
        }
        if (normalized === "disabled") {
            return "Disabled";
        }
        if (normalized === "omitted") {
            return "Omitted";
        }
        return normalized ? normalized.replace(/_/g, " ") : "--";
    }

    function humanizeReason(reason) {
        const raw = String(reason || "").trim();
        if (!raw) {
            return "The current setup does not clear the minimum edge needed to act.";
        }
        if (directReasonMap[raw]) {
            return directReasonMap[raw];
        }
        if (raw.includes("does not pass the live trading guardrails")) {
            return "This setup does not clear the live trading guardrails.";
        }
        if (raw.includes("overpaying")) {
            return "The market looks too expensive relative to the model, so the value sits on the other side.";
        }
        if (raw.includes("main scenario")) {
            return "This remains the model's main scenario.";
        }
        if (raw.includes("discounted enough to buy")) {
            return "The price is discounted enough versus the model to consider a position.";
        }
        if (raw.includes("next official print comes in hotter")) {
            return "This path stays alive if the next official print comes in hotter.";
        }
        if (raw.includes("remaining upside")) {
            return "There is still enough room left in today's path for this scenario.";
        }
        if (raw.includes("Still inside the model range")) {
            return "The setup remains inside the model's expected range.";
        }
        return raw;
    }

    function compareModelAndMarket(summary) {
        const modelLabel = bracketLabel(summary?.model?.top_label, "");
        const marketLabel = bracketLabel(summary?.market?.top_label, "");
        if (!modelLabel || !marketLabel) {
            return "Market and model alignment is still forming.";
        }
        if (modelLabel === marketLabel) {
            return "Market and model are pointing to the same bracket.";
        }
        return "Market and model are leaning to different brackets.";
    }

    function realitySupport(summary) {
        const modelLabel = String(summary?.model?.top_label || "");
        const currentReality = String(summary?.reality?.current_reality_bracket || "");
        if (!currentReality) {
            if (Number(state.targetDay) === 0) {
                return "Live reality has not locked onto a bracket yet.";
            }
            return "This horizon is forecast-driven rather than live-observation driven.";
        }
        if (currentReality === modelLabel) {
            return "Current reality supports the model's top scenario so far.";
        }
        return "Current reality is tracking a different bracket so far.";
    }

    function activeNextOfficial(detail) {
        return detail?.tactical?.next_metar || detail?.reality?.next_official || null;
    }

    function buildWhyBullets(detail) {
        const summary = detail?.summary || {};
        const bullets = [];
        const sourceRows = detail?.source_detail || [];
        const okSources = sourceRows.filter((row) => String(row?.status || "").toLowerCase() === "ok").length;
        const spread = summary?.model?.source_spread;

        if (Number(detail?.target_day) === 0) {
            if (summary?.reality?.official_temp_market != null) {
                bullets.push("Intraday mode is anchored on live official observations and PWS context rather than future source strips.");
            } else {
                bullets.push("Intraday mode is waiting for live official observations to firm up reality.");
            }
        } else if (!sourceRows.length) {
            bullets.push("Future-source snapshots are not available yet for this horizon.");
        } else if (okSources <= 1) {
            bullets.push("Only one forecast source is carrying the view right now, so conviction is limited.");
        } else if (typeof spread === "number" && spread <= 1.0) {
            bullets.push("Forecast sources are closely aligned, which supports the current model winner.");
        } else if (typeof spread === "number" && spread >= 2.5) {
            bullets.push("Forecast sources are pulling apart, so the top scenario has more uncertainty than usual.");
        } else {
            bullets.push(`${okSources} of ${sourceRows.length} tracked forecast sources are usable for this horizon.`);
        }

        if (summary?.model?.calibration_active) {
            bullets.push("Recent resolved days are actively reweighting the source blend.");
        } else if (summary?.model?.calibration_warming_up) {
            bullets.push("Calibration is still warming up, so weights stay close to the fixed baseline.");
        } else if ((summary?.model?.calibration_samples || 0) > 0) {
            bullets.push("Calibration is being tracked, but it is not materially changing the blend yet.");
        } else {
            bullets.push("There is not enough resolved history yet to calibrate this horizon.");
        }

        bullets.push(humanizeReason(summary?.actionable?.reason));

        const nextOfficial = activeNextOfficial(detail);
        const direction = String(nextOfficial?.direction || "").toUpperCase();
        if (direction === "UP") {
            bullets.push("The next official print is projected higher, which keeps upside paths alive.");
        } else if (direction === "DOWN") {
            bullets.push("The next official print is projected lower, which caps late upside trades.");
        } else if (direction === "FLAT") {
            bullets.push("The next official print is not expected to change the picture much.");
        } else if (summary?.reality?.official_temp_market != null) {
            bullets.push(`Official observations are live, with a running day max near ${formatNumber(summary.reality.cumulative_max_market, 1)}.`);
        }

        return [...new Set(bullets.filter(Boolean))].slice(0, 4);
    }

    function renderStationContext(detail = state.payload) {
        const stationId = state.stationId || currentStationId();
        const option = currentStationOption(stationId);
        const summary = detail?.summary || {};
        const targetDate = detail?.target_date || summary?.target_date || null;
        const marketStatus = summary?.market?.status === "CLOSED" ? "Market closed" : "Market open";
        const contextChips = [
            `<span class="pl-chip">${esc(horizonLabel())}</span>`,
            `<span class="pl-chip">${esc(option?.city_name || stationId)}</span>`,
            option?.market_unit ? `<span class="pl-chip">${esc(String(option.market_unit).toUpperCase())} market</span>` : "",
            targetDate ? `<span class="pl-chip">${esc(formatDate(targetDate))}</span>` : "",
            summary?.market?.status ? `<span class="pl-chip">${esc(marketStatus)}</span>` : "",
        ].filter(Boolean);

        els.contextChips.innerHTML = contextChips.join("");
        els.stationTitle.textContent = option?.label || stationId;
        els.stationMeta.textContent = [
            option?.name || "",
            option?.timezone || "",
        ].filter(Boolean).join(" | ") || `Station ${stationId}`;
        els.stationContext.textContent = [
            horizonLabel(),
            targetDate ? formatDate(targetDate) : "",
            summary?.market?.event_title || "",
        ].filter(Boolean).join(" | ") || "Loading the selected horizon.";
    }

    function renderSummaryCard(element, label, value, note, footnote) {
        element.innerHTML = `
            <div class="pl-section-label">${esc(label)}</div>
            <div class="pl-summary-value">${esc(value)}</div>
            <div class="pl-note pl-summary-note">${esc(note)}</div>
            ${footnote ? `<div class="pl-note pl-summary-note">${esc(footnote)}</div>` : ""}
        `;
    }

    function buildActionPresenter(detail) {
        const summary = detail?.summary || {};
        const actionable = summary?.actionable || {};
        const label = bracketLabel(actionable?.label || summary?.model?.top_label);
        const confidence = formatPercent(summary?.model?.confidence, 0);
        const entry = formatPercent(actionable?.entry_price, 0);
        const fair = formatPercent(actionable?.fair_prob, 0);
        const edge = typeof actionable?.edge_points === "number" ? `${actionable.edge_points.toFixed(1)} pts edge` : null;

        if (actionable?.available && String(actionable?.side || "").toUpperCase() === "YES") {
            return {
                tone: "buy-yes",
                badge: "BUY YES",
                title: `Buy YES on ${label}`,
                copy: humanizeReason(actionable?.reason),
                meta: [edge, fair !== "--" ? `Fair value ${fair}` : null, entry !== "--" ? `Entry price ${entry}` : null, confidence !== "--" ? `Model confidence ${confidence}` : null].filter(Boolean),
            };
        }

        if (actionable?.available && String(actionable?.side || "").toUpperCase() === "NO") {
            return {
                tone: "buy-no",
                badge: "BUY NO",
                title: `Buy NO on ${label}`,
                copy: humanizeReason(actionable?.reason),
                meta: [edge, fair !== "--" ? `Fair value ${fair}` : null, entry !== "--" ? `Entry price ${entry}` : null, confidence !== "--" ? `Model confidence ${confidence}` : null].filter(Boolean),
            };
        }

        return {
            tone: "no-trade",
            badge: "NO TRADE",
            title: "No trade right now",
            copy: humanizeReason(actionable?.reason),
            meta: [
                summary?.model?.top_label ? `Model winner ${bracketLabel(summary.model.top_label)}` : null,
                confidence !== "--" ? `Model confidence ${confidence}` : null,
                summary?.market?.top_label ? `Market favorite ${bracketLabel(summary.market.top_label)}` : null,
                compareModelAndMarket(summary),
            ].filter(Boolean),
        };
    }

    function renderPrimary(detail) {
        const presenter = buildActionPresenter(detail);
        els.actionBadge.className = `pl-action-badge ${presenter.tone}`;
        els.actionBadge.textContent = presenter.badge;
        els.actionTitle.textContent = presenter.title;
        els.actionCopy.textContent = presenter.copy;
        els.actionMeta.innerHTML = presenter.meta.length
            ? presenter.meta.map((item) => `<span class="pl-primary-meta-item">${esc(item)}</span>`).join("")
            : '<span class="pl-primary-meta-item">Waiting for a clearer setup</span>';
    }

    function renderSummary(detail) {
        const summary = detail?.summary || {};
        const modelLabel = bracketLabel(summary?.model?.top_label);
        const marketLabel = bracketLabel(summary?.market?.top_label, "No clear market bracket yet");
        const currentReality = bracketLabel(summary?.reality?.current_reality_bracket || summary?.reality?.resolved_winning_bracket, Number(detail?.target_day) === 0 ? "No live bracket yet" : "No live reality bracket");

        renderSummaryCard(
            els.likelyCard,
            "Most likely outcome",
            modelLabel,
            summary?.model?.confidence != null ? `Model confidence ${formatPercent(summary.model.confidence, 0)}` : "The model has not formed a confident bracket yet.",
            summary?.model?.mean != null && summary?.model?.sigma != null ? `Centered near ${formatNumber(summary.model.mean, 1)} with ${formatNumber(summary.model.sigma, 2)} spread.` : ""
        );

        renderSummaryCard(
            els.marketCard,
            "What the market says",
            marketLabel,
            summary?.market?.top_price != null ? `Top YES price ${formatPercent(summary.market.top_price, 0)}.` : "Market pricing is not available yet.",
            summary?.market?.volume != null ? `${compareModelAndMarket(summary)} Volume ${formatMoney(summary.market.volume)}.` : compareModelAndMarket(summary)
        );

        const realityNote = summary?.reality?.official_temp_market != null
            ? `Official ${formatNumber(summary.reality.official_temp_market, 1)} | running max ${formatNumber(summary.reality.cumulative_max_market, 1)}`
            : (Number(detail?.target_day) === 0 ? "Waiting for enough official observations to define reality." : "This horizon is driven by forecast sources, not live observations.");

        renderSummaryCard(
            els.realityCard,
            "Current reality",
            currentReality,
            realityNote,
            realitySupport(summary)
        );
    }

    function renderWhy(detail) {
        const bullets = buildWhyBullets(detail);
        els.whyList.innerHTML = bullets.length
            ? bullets.map((item) => `<li>${esc(item)}</li>`).join("")
            : "<li>Waiting for a stable explanation.</li>";
    }

    function renderLoading() {
        destroyChart();
        renderStationContext(null);
        els.actionBadge.className = "pl-action-badge no-trade";
        els.actionBadge.textContent = "Checking";
        els.actionTitle.textContent = "Loading recommendation...";
        els.actionCopy.textContent = "Pulling the latest model, market, and reality context for the active station.";
        els.actionMeta.innerHTML = '<span class="pl-primary-meta-item">Refreshing station view</span>';
        renderSummaryCard(els.likelyCard, "Most likely outcome", "--", "Waiting for the model.", "");
        renderSummaryCard(els.marketCard, "What the market says", "--", "Waiting for the market snapshot.", "");
        renderSummaryCard(els.realityCard, "Current reality", "--", "Waiting for live context.", "");
        els.whyList.innerHTML = "<li>Collecting the latest reasons behind the current recommendation.</li>";
        els.advancedSummary.textContent = "External-source charts, the full YES/NO ladder, calibration, and PWS context for the active station.";
        els.historyMeta.textContent = els.advancedDetails.open
            ? "Loading source history for the active horizon."
            : "Open the technical detail to load source history.";
        els.sourceList.innerHTML = '<div class="pl-loading">Waiting for source detail.</div>';
        els.ladder.innerHTML = '<div class="pl-loading">Waiting for bracket details.</div>';
        els.realityDetails.innerHTML = '<div class="pl-loading">Waiting for reality detail.</div>';
        els.calibration.innerHTML = '<div class="pl-loading">Waiting for calibration detail.</div>';
        els.pws.innerHTML = '<div class="pl-loading">Waiting for PWS detail.</div>';
    }

    function renderError(error) {
        destroyChart();
        const message = error && error.message ? error.message : String(error || "Unknown error");
        renderStationContext(null);
        els.actionBadge.className = "pl-action-badge no-trade";
        els.actionBadge.textContent = "Error";
        els.actionTitle.textContent = "Probability Lab could not load this station";
        els.actionCopy.textContent = message;
        els.actionMeta.innerHTML = '<span class="pl-primary-meta-item">Check the station payload and try again</span>';
        els.likelyCard.innerHTML = `<div class="pl-error">${esc(message)}</div>`;
        els.marketCard.innerHTML = '<div class="pl-empty">Market context is unavailable.</div>';
        els.realityCard.innerHTML = '<div class="pl-empty">Reality context is unavailable.</div>';
        els.whyList.innerHTML = "<li>The page could not assemble a usable summary from the current payload.</li>";
        els.advancedSummary.textContent = "Technical detail is unavailable until the station payload loads successfully.";
        els.historyMeta.textContent = "History could not be loaded.";
        els.sourceList.innerHTML = `<div class="pl-error">${esc(message)}</div>`;
        els.ladder.innerHTML = '<div class="pl-empty">No ladder is available right now.</div>';
        els.realityDetails.innerHTML = '<div class="pl-empty">No reality detail is available right now.</div>';
        els.calibration.innerHTML = '<div class="pl-empty">No calibration detail is available right now.</div>';
        els.pws.innerHTML = '<div class="pl-empty">No PWS detail is available right now.</div>';
    }

    function renderSourceList(detail) {
        const rows = detail?.source_detail || [];
        if (!rows.length) {
            els.sourceList.innerHTML = '<div class="pl-empty">No source snapshots are available for this horizon.</div>';
            return;
        }

        els.sourceList.innerHTML = rows.map((row) => {
            const status = String(row?.status || "").toLowerCase();
            const notes = Array.isArray(row?.notes) ? row.notes.filter(Boolean) : [];
            return `
                <div class="pl-source-card">
                    <div class="pl-source-head">
                        <strong>${esc(sourceName(row?.source))}</strong>
                        <span class="pl-source-status ${esc(status)}">${esc(sourceStatusText(row?.status))}</span>
                    </div>
                    <div class="pl-note">High ${esc(formatNumber(row?.forecast_high_market, 1))}</div>
                    <div class="pl-note">${row?.peak_hour_local != null ? `Peak near ${esc(formatNumber(Number(row.peak_hour_local), 1))}h local.` : "Peak timing not available."}</div>
                    <div class="pl-note">Updated ${esc(formatStamp(row?.provider_updated_local || row?.captured_at_utc))}</div>
                    ${notes.length ? `<div class="pl-note">${esc(notes.join(" | "))}</div>` : ""}
                </div>
            `;
        }).join("");
    }

    function renderLadder(detail) {
        const rows = detail?.bracket_ladder || [];
        if (!rows.length) {
            els.ladder.innerHTML = '<div class="pl-empty">No bracket ladder is available for this station.</div>';
            return;
        }

        els.ladder.innerHTML = `
            <div class="pl-table-wrap">
            <table class="pl-table">
                <thead>
                    <tr>
                        <th>Bracket</th>
                        <th>Fair YES</th>
                        <th>Fair NO</th>
                        <th>Market YES</th>
                        <th>Market NO</th>
                        <th>Active side</th>
                        <th>Fair / entry</th>
                        <th>Edge</th>
                        <th>Recommendation</th>
                        <th>Why</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows.map((row) => {
                        const selectedSide = String(row?.active_selected_side || row?.selected_side || "--").toUpperCase();
                        const fair = row?.active_selected_fair ?? row?.selected_fair;
                        const entry = row?.active_selected_entry ?? row?.selected_entry;
                        const edgePoints = row?.active_edge_points ?? row?.edge_points;
                        const bestEdge = row?.active_best_edge ?? row?.best_edge;
                        const recommendation = recommendationText(row?.active_recommendation || row?.recommendation);
                        const why = humanizeReason(row?.active_policy_reason || row?.policy_reason);
                        return `
                            <tr>
                                <td><strong>${esc(bracketLabel(row?.label))}</strong></td>
                                <td>${esc(formatPercent(row?.fair_yes, 0))}</td>
                                <td>${esc(formatPercent(row?.fair_no, 0))}</td>
                                <td>${esc(formatPercent(row?.market_yes, 0))}</td>
                                <td>${esc(formatPercent(row?.market_no, 0))}</td>
                                <td>${esc(selectedSide)}</td>
                                <td>${esc(formatPercent(fair, 0))} fair | ${esc(formatPercent(entry, 0))} entry</td>
                                <td>${typeof edgePoints === "number" ? `${edgePoints.toFixed(1)} pts` : "--"}${typeof bestEdge === "number" ? ` | ${formatPercent(bestEdge, 1)} raw` : ""}</td>
                                <td>${esc(recommendation)}</td>
                                <td>${esc(why)}</td>
                            </tr>
                        `;
                    }).join("")}
                </tbody>
            </table>
            </div>
        `;
    }

    function formatRealityItems(detail) {
        const rows = [];
        const reality = detail?.reality || {};

        Object.entries(reality).forEach(([key, value]) => {
            if (value == null || key === "next_official") {
                return;
            }
            rows.push({
                label: realityLabels[key] || key.replace(/_/g, " "),
                value: typeof value === "number" ? formatNumber(value, Math.abs(value) >= 10 ? 1 : 2) : String(value),
            });
        });

        const nextOfficial = activeNextOfficial(detail);
        if (nextOfficial?.direction) {
            rows.push({ label: "Next official direction", value: String(nextOfficial.direction).toUpperCase() });
        }
        if (nextOfficial?.minutes_to_next != null) {
            rows.push({ label: "Minutes to next print", value: `${formatNumber(Number(nextOfficial.minutes_to_next), 0)} min` });
        }
        if (nextOfficial?.next_obs_utc) {
            rows.push({ label: "Next official print", value: formatStamp(nextOfficial.next_obs_utc) });
        }

        return rows;
    }

    function renderRealityDetails(detail) {
        const rows = formatRealityItems(detail);
        if (!rows.length) {
            els.realityDetails.innerHTML = '<div class="pl-empty">No reality detail is available for this horizon.</div>';
            return;
        }

        els.realityDetails.innerHTML = rows.map((row) => {
            return `
                <div class="pl-data-list-item">
                    <div class="pl-mini-label">${esc(row.label)}</div>
                    <div class="pl-detail-value">${esc(row.value)}</div>
                </div>
            `;
        }).join("");
    }

    function calibrationStatusText(calibration) {
        if (calibration?.active) {
            return "Actively reweighting sources";
        }
        if (calibration?.warming_up) {
            return "Warming up";
        }
        if ((calibration?.samples || 0) > 0) {
            return "Tracked only";
        }
        return "Not available yet";
    }

    function renderCalibration(detail) {
        const calibration = detail?.calibration || {};
        const ranking = calibration?.source_ranking || [];
        const appliedSources = (calibration?.applied_sources || []).map(sourceName);

        const summaryHtml = `
            <div class="pl-data-list">
                <div class="pl-data-list-item">
                    <div class="pl-mini-label">Status</div>
                    <div class="pl-detail-value">${esc(calibrationStatusText(calibration))}</div>
                </div>
                <div class="pl-data-list-item">
                    <div class="pl-mini-label">Samples</div>
                    <div class="pl-detail-value">${esc(String(calibration?.samples ?? 0))}</div>
                </div>
                <div class="pl-data-list-item">
                    <div class="pl-mini-label">Model MAE</div>
                    <div class="pl-detail-value">${esc(formatNumber(calibration?.mae_market, 2))}</div>
                </div>
                <div class="pl-data-list-item">
                    <div class="pl-mini-label">Model bias</div>
                    <div class="pl-detail-value">${esc(formatNumber(calibration?.bias_market, 2))}</div>
                </div>
            </div>
            <div class="pl-note" style="margin-top: 14px;">
                ${appliedSources.length ? `Applied sources: ${esc(appliedSources.join(", "))}.` : "No source adjustments are being applied right now."}
            </div>
        `;

        if (!ranking.length) {
            els.calibration.innerHTML = `${summaryHtml}<div class="pl-empty" style="margin-top: 14px;">No source ranking is available yet.</div>`;
            return;
        }

        els.calibration.innerHTML = `
            ${summaryHtml}
            <table class="pl-table compact">
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>Samples</th>
                        <th>MAE</th>
                        <th>Bias</th>
                        <th>Multiplier</th>
                    </tr>
                </thead>
                <tbody>
                    ${ranking.map((row) => {
                        return `
                            <tr>
                                <td><strong>${esc(sourceName(row?.source))}</strong></td>
                                <td>${esc(String(row?.samples ?? 0))}</td>
                                <td>${esc(formatNumber(row?.mae_market, 2))}</td>
                                <td>${esc(formatNumber(row?.bias_market, 2))}</td>
                                <td>${esc(formatNumber(row?.calibration_multiplier, 2))}</td>
                            </tr>
                        `;
                    }).join("")}
                </tbody>
            </table>
        `;
    }

    function renderPws(detail) {
        const rows = detail?.pws_profiles || [];
        if (!rows.length) {
            els.pws.innerHTML = Number(detail?.target_day) === 0
                ? '<div class="pl-empty">No predictive PWS profiles are available right now.</div>'
                : '<div class="pl-empty">PWS profiles are only populated for the same-day view.</div>';
            return;
        }

        els.pws.innerHTML = rows.map((row) => {
            return `
                <div class="pl-source-card">
                    <strong>${esc(row?.station_id || "--")}</strong>
                    <div class="pl-note">Next METAR score ${esc(formatNumber(row?.next_metar_score, 1))}</div>
                    <div class="pl-note">Eligible ${row?.rank_eligible ? "yes" : "no"} | weight ${esc(formatNumber(row?.weight || row?.learning_weight_predictive, 2))}</div>
                    <div class="pl-note">Age ${esc(formatNumber(row?.age_minutes, 1))} min</div>
                </div>
            `;
        }).join("");
    }

    function renderHistory(detail) {
        if (!els.advancedDetails.open) {
            destroyChart();
            els.historyMeta.textContent = "Open the technical detail to load source history.";
            return;
        }

        destroyChart();
        const series = detail?.source_history?.series || [];

        if (!series.length) {
            els.historyMeta.textContent = detail?.source_history?.history_warming_up
                ? "History is still warming up. Fresh snapshots exist, but the lookback is still shallow."
                : "No source history is available for this horizon.";
            return;
        }

        els.historyMeta.textContent = `${series.length} source series over the requested lookback window.`;

        const stampSet = new Set();
        series.forEach((row) => {
            (row?.points || []).forEach((point) => {
                if (point?.captured_at_utc) {
                    stampSet.add(point.captured_at_utc);
                }
            });
        });

        const labels = [...stampSet].sort((left, right) => new Date(left) - new Date(right));
        const datasets = series.map((row) => {
            const pointMap = new Map((row?.points || []).map((point) => [point.captured_at_utc, point.forecast_high_market]));
            return {
                label: sourceName(row?.source),
                data: labels.map((label) => (pointMap.has(label) ? pointMap.get(label) : null)),
                borderColor: sourceColors[row?.source] || "#cbd5e1",
                backgroundColor: "transparent",
                tension: 0.28,
                spanGaps: true,
                pointRadius: 2,
                borderWidth: 2,
            };
        });

        state.chart = new Chart(els.historyChart.getContext("2d"), {
            type: "line",
            data: {
                labels: labels.map(formatStamp),
                datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "nearest",
                    intersect: false,
                },
                plugins: {
                    legend: {
                        labels: {
                            color: "#cbd5e1",
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: {
                            color: "#94a3b8",
                            autoSkip: true,
                            maxRotation: 0,
                            maxTicksLimit: 8,
                        },
                        grid: {
                            color: "rgba(148,163,184,0.12)",
                        },
                    },
                    y: {
                        ticks: {
                            color: "#94a3b8",
                        },
                        grid: {
                            color: "rgba(148,163,184,0.12)",
                        },
                    },
                },
            },
        });
    }

    function renderAdvanced(detail) {
        renderSourceList(detail);
        renderLadder(detail);
        renderRealityDetails(detail);
        renderCalibration(detail);
        renderPws(detail);
        els.advancedSummary.textContent = `External-source charts, the full YES/NO ladder, calibration, and PWS context for ${currentStationOption()?.label || detail?.station_id || state.stationId}.`;
        renderHistory(detail);
    }

    function renderPayload(detail) {
        state.payload = detail;
        renderStationContext(detail);
        renderPrimary(detail);
        renderSummary(detail);
        renderWhy(detail);
        renderAdvanced(detail);
    }

    function updateToggleButtons() {
        els.toggle.querySelectorAll("button").forEach((button) => {
            button.classList.toggle("active", Number(button.dataset.day) === Number(state.targetDay));
        });
    }

    async function loadProbabilityLab() {
        state.stationId = currentStationId();
        const requestId = ++state.requestId;
        renderLoading();
        updateToggleButtons();

        try {
            const detail = await fetchJSON(`/api/probability-lab/station/${encodeURIComponent(state.stationId)}?target_day=${state.targetDay}&lookback_hours=36`);
            if (requestId !== state.requestId) {
                return;
            }
            renderPayload(detail);
        } catch (error) {
            if (requestId !== state.requestId) {
                return;
            }
            renderError(error);
        }
    }

    els.toggle.querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => {
            const nextDay = Number(button.dataset.day);
            if (Number.isNaN(nextDay) || nextDay === state.targetDay) {
                return;
            }
            state.targetDay = nextDay;
            loadProbabilityLab();
        });
    });

    els.advancedDetails.addEventListener("toggle", () => {
        if (!state.payload) {
            return;
        }
        renderHistory(state.payload);
    });

    window.addEventListener("helios:stationchange", (event) => {
        const nextStation = String(event?.detail?.stationId || currentStationId()).toUpperCase();
        if (nextStation === state.stationId && state.payload) {
            renderStationContext(state.payload);
            return;
        }
        state.stationId = nextStation;
        loadProbabilityLab();
    });

    loadProbabilityLab();
})();
