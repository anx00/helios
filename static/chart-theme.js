/**
 * HELIOS Chart Theme Integration
 * Reads CSS custom properties and applies them to Chart.js defaults.
 * Listens for theme changes and updates all registered charts.
 */
(function () {
    'use strict';

    /** Registry of active Chart.js instances */
    window.heliosCharts = window.heliosCharts || [];

    function getCSSVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    /** Returns the current chart palette derived from CSS tokens */
    function getChartPalette() {
        return {
            grid: getCSSVar('--chart-grid') || 'rgba(255,255,255,0.06)',
            text: getCSSVar('--chart-text') || '#94a3b8',
            colors: [
                getCSSVar('--chart-color-1') || '#3b82f6',
                getCSSVar('--chart-color-2') || '#10b981',
                getCSSVar('--chart-color-3') || '#f59e0b',
                getCSSVar('--chart-color-4') || '#8b5cf6',
                getCSSVar('--chart-color-5') || '#ef4444',
                getCSSVar('--chart-color-6') || '#06b6d4',
            ],
            bg: getCSSVar('--color-bg-base') || '#0a0e17',
            surface: getCSSVar('--color-surface-1') || 'rgba(255,255,255,0.03)',
            border: getCSSVar('--color-border-subtle') || 'rgba(255,255,255,0.06)',
            textPrimary: getCSSVar('--color-text-primary') || '#f1f5f9',
            textSecondary: getCSSVar('--color-text-secondary') || '#94a3b8',
            accent: getCSSVar('--color-accent-blue') || '#3b82f6',
            success: getCSSVar('--color-accent-green') || '#10b981',
            warning: getCSSVar('--color-accent-yellow') || '#f59e0b',
            danger: getCSSVar('--color-accent-red') || '#ef4444',
        };
    }

    /** Apply palette to Chart.js global defaults */
    function applyChartDefaults() {
        if (typeof Chart === 'undefined') return;

        const palette = getChartPalette();

        Chart.defaults.color = palette.text;
        Chart.defaults.borderColor = palette.grid;

        // Scale defaults
        if (Chart.defaults.scales) {
            const scaleTypes = ['linear', 'category', 'time', 'logarithmic'];
            scaleTypes.forEach(function (type) {
                if (Chart.defaults.scales[type]) {
                    Chart.defaults.scales[type].grid = Chart.defaults.scales[type].grid || {};
                    Chart.defaults.scales[type].grid.color = palette.grid;
                    Chart.defaults.scales[type].ticks = Chart.defaults.scales[type].ticks || {};
                    Chart.defaults.scales[type].ticks.color = palette.text;
                }
            });
        }

        // Plugin defaults
        if (Chart.defaults.plugins) {
            if (Chart.defaults.plugins.legend) {
                Chart.defaults.plugins.legend.labels = Chart.defaults.plugins.legend.labels || {};
                Chart.defaults.plugins.legend.labels.color = palette.text;
            }
            if (Chart.defaults.plugins.title) {
                Chart.defaults.plugins.title.color = palette.textPrimary;
            }
            if (Chart.defaults.plugins.tooltip) {
                Chart.defaults.plugins.tooltip.backgroundColor = palette.bg;
                Chart.defaults.plugins.tooltip.titleColor = palette.textPrimary;
                Chart.defaults.plugins.tooltip.bodyColor = palette.textSecondary;
                Chart.defaults.plugins.tooltip.borderColor = palette.border;
                Chart.defaults.plugins.tooltip.borderWidth = 1;
            }
        }
    }

    /** Update all registered charts when theme changes */
    function updateAllCharts() {
        applyChartDefaults();

        var charts = window.heliosCharts || [];
        for (var i = charts.length - 1; i >= 0; i--) {
            var chart = charts[i];
            if (!chart || !chart.canvas || !chart.canvas.isConnected) {
                // Remove destroyed/detached charts
                charts.splice(i, 1);
                continue;
            }
            try {
                // Update scale colors
                var palette = getChartPalette();
                Object.keys(chart.scales || {}).forEach(function (key) {
                    var scale = chart.scales[key];
                    if (scale.options.grid) {
                        scale.options.grid.color = palette.grid;
                    }
                    if (scale.options.ticks) {
                        scale.options.ticks.color = palette.text;
                    }
                });
                // Update legend
                if (chart.options.plugins && chart.options.plugins.legend && chart.options.plugins.legend.labels) {
                    chart.options.plugins.legend.labels.color = palette.text;
                }
                chart.update('none');
            } catch (_) {
                charts.splice(i, 1);
            }
        }
    }

    /** Register a chart for automatic theme updates */
    function registerChart(chart) {
        if (chart && window.heliosCharts.indexOf(chart) === -1) {
            window.heliosCharts.push(chart);
        }
        return chart;
    }

    // Initialize
    function init() {
        applyChartDefaults();

        // Listen for theme changes
        window.addEventListener('helios:theme-changed', function () {
            // Small delay to let CSS variables propagate
            requestAnimationFrame(function () {
                updateAllCharts();
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }

    // Public API
    window.HeliosChartTheme = {
        getPalette: getChartPalette,
        applyDefaults: applyChartDefaults,
        register: registerChart,
        updateAll: updateAllCharts,
    };
})();
