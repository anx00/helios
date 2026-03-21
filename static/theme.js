(function () {
    const STORAGE_KEY = 'heliosTheme';
    const THEMES = new Set(['dark', 'light']);

    function getPreferredTheme() {
        if (typeof window === 'undefined') return 'dark';
        try {
            const stored = window.localStorage.getItem(STORAGE_KEY);
            if (THEMES.has(stored)) return stored;
        } catch (_) {}

        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
            return 'light';
        }
        return 'dark';
    }

    function setTheme(theme, persist) {
        const nextTheme = THEMES.has(theme) ? theme : 'dark';
        const root = document.documentElement;
        root.dataset.theme = nextTheme;
        root.style.colorScheme = nextTheme;
        if (document.body) {
            document.body.dataset.theme = nextTheme;
        }

        document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
            const icon = button.querySelector('[data-theme-icon]');
            const label = button.querySelector('[data-theme-label]');
            const isDark = nextTheme === 'dark';
            const nextLabel = isDark ? 'Light mode' : 'Dark mode';

            button.setAttribute('aria-label', `Switch to ${isDark ? 'light' : 'dark'} mode`);
            button.title = `Switch to ${isDark ? 'light' : 'dark'} mode`;
            button.dataset.theme = nextTheme;

            if (icon) {
                icon.setAttribute('data-lucide', isDark ? 'sun' : 'moon');
            }
            if (label) {
                label.textContent = nextLabel;
            }
        });

        if (persist) {
            try {
                window.localStorage.setItem(STORAGE_KEY, nextTheme);
            } catch (_) {}
        }

        if (window.lucide && typeof window.lucide.createIcons === 'function') {
            window.lucide.createIcons();
        }

        // Dispatch theme change event for chart-theme.js and other listeners
        try {
            window.dispatchEvent(new CustomEvent('helios:theme-changed', {
                detail: { theme: nextTheme }
            }));
        } catch (_) {}

        return nextTheme;
    }

    function toggleTheme() {
        const currentTheme = document.documentElement.dataset.theme || getPreferredTheme();
        return setTheme(currentTheme === 'dark' ? 'light' : 'dark', true);
    }

    function init() {
        setTheme(getPreferredTheme(), false);
        window.toggleTheme = toggleTheme;

        window.addEventListener('storage', (event) => {
            if (event.key === STORAGE_KEY && THEMES.has(event.newValue)) {
                setTheme(event.newValue, false);
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }

    window.HeliosTheme = {
        get() {
            return document.documentElement.dataset.theme || getPreferredTheme();
        },
        set(theme) {
            return setTheme(theme, true);
        },
        toggle: toggleTheme,
    };
})();
