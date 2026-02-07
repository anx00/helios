
// Station IDs - fetched dynamically from backend
let STATIONS = [];

async function loadStations() {
    try {
        const resp = await fetch('/api/stations');
        const data = await resp.json();
        STATIONS = data.map(s => s.id);
    } catch (e) {
        console.error('Failed to load stations, using fallback:', e);
        STATIONS = ['KLGA', 'KATL'];
    }
}

async function fetchPrediction(stationId) {
    try {
        const response = await fetch(`/api/prediction/${stationId}`);
        const data = await response.json();

        if (data.error) {
            console.warn(`No data for ${stationId}`);
            updateBadge(stationId, null);
            return;
        }

        // Update main temp
        const tempEl = document.getElementById(`pred-${stationId}`);
        if (tempEl) tempEl.textContent = data.prediction_f.toFixed(1);

        // Update Badge (Today vs Tomorrow)
        updateBadge(stationId, data.timestamp);

    } catch (e) {
        console.error(`Error fetching ${stationId}:`, e);
    }
}

function updateBadge(stationId, timestamp) {
    const badge = document.getElementById(`badge-${stationId}`);
    if (!badge) return;

    if (!timestamp) {
        badge.textContent = '-';
        return;
    }

    const predictionDate = new Date(timestamp);
    const today = new Date();

    // Check if prediction is for today or tomorrow
    // Warning: 'timestamp' in DB is creation time.
    // If created recently (e.g. within last 20 hours), it's likely for the active market.
    // Logic: compare Day of Month.
    // This is heuristic because we don't return 'target_date' from API yet, but good enough for now.

    // Actually, let's assume if it matches today's date, it's TODAY.
    const isToday = predictionDate.getDate() === today.getDate() &&
        predictionDate.getMonth() === today.getMonth();

    if (isToday) {
        badge.textContent = 'TODAY';
        badge.classList.add('today');
        badge.classList.remove('tomorrow');
    } else {
        badge.textContent = 'TOMORROW';
        badge.classList.add('tomorrow');
        badge.classList.remove('today');
    }
}

async function refreshStation(stationId, event) {
    // Stop propagation so card doesn't toggle
    event.stopPropagation();

    const btn = document.getElementById(`refresh-${stationId}`);
    if (btn) btn.classList.add('spinning');

    try {
        const response = await fetch(`/api/refresh/${stationId}`, { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            // Re-fetch data
            await fetchPrediction(stationId);
            await fetchMarketData(stationId);
        } else {
            alert('Update failed: ' + (result.error || 'Unknown error'));
        }
    } catch (e) {
        alert('Error connecting to backend');
    } finally {
        if (btn) btn.classList.remove('spinning');
    }
}

function toggleCard(stationId) {
    const card = document.getElementById(`card-${stationId}`);
    if (card) {
        card.classList.toggle('expanded');
    }
}

// ============================================
// POLYMARKET WIDGET
// ============================================

async function fetchMarketData(stationId) {
    try {
        const response = await fetch(`/api/market/${stationId}`);
        const data = await response.json();

        if (data.error) {
            console.warn(`No market for ${stationId}: ${data.error}`);
            return;
        }

        // Build legend (top 3 outcomes for compact index view)
        const legendEl = document.getElementById(`market-legend-${stationId}`);
        if (legendEl && data.outcomes) {
            legendEl.innerHTML = '';
            const topOutcomes = data.outcomes.slice(0, 3);
            topOutcomes.forEach((outcome, idx) => {
                const item = document.createElement('div');
                item.className = `legend-item color-${idx + 1}`;
                item.innerHTML = `
                    <span class="legend-dot"></span>
                    <span>${outcome.title} ${(outcome.probability * 100).toFixed(0)}%</span>
                `;
                legendEl.appendChild(item);
            });
        }

    } catch (e) {
        console.error(`Error fetching market for ${stationId}:`, e);
    }
}

async function init() {
    await loadStations();
    STATIONS.forEach(id => {
        fetchPrediction(id);
        fetchMarketData(id);
    });

    // Refresh every 5 minutes
    setInterval(() => {
        STATIONS.forEach(id => {
            fetchPrediction(id);
            fetchMarketData(id);
        });
    }, 5 * 60 * 1000);
}

document.addEventListener('DOMContentLoaded', init);
