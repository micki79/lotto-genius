// LottoGenius Service Worker
const CACHE_NAME = 'lottogenius-v3.17.0';
const DATA_CACHE_NAME = 'lottogenius-data-v3.17.0';

// Assets to cache immediately
const STATIC_ASSETS = [
    './',
    './index.html',
    './manifest.json',
    'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('[SW] Installing...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[SW] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => {
                console.log('[SW] Installation complete');
                return self.skipWaiting();
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating...');
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => {
                            return name !== CACHE_NAME && name !== DATA_CACHE_NAME;
                        })
                        .map((name) => {
                            console.log('[SW] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => {
                console.log('[SW] Activation complete');
                return self.clients.claim();
            })
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const requestUrl = new URL(event.request.url);

    // Handle API requests (lottery data)
    if (requestUrl.hostname === 'johannesfriedrich.github.io') {
        event.respondWith(
            caches.open(DATA_CACHE_NAME)
                .then((cache) => {
                    return fetch(event.request)
                        .then((response) => {
                            // If response is valid, update cache
                            if (response.status === 200) {
                                cache.put(event.request, response.clone());
                            }
                            return response;
                        })
                        .catch(() => {
                            // Network failed, try cache
                            return cache.match(event.request);
                        });
                })
        );
        return;
    }

    // Handle Google Fonts
    if (requestUrl.hostname === 'fonts.googleapis.com' ||
        requestUrl.hostname === 'fonts.gstatic.com') {
        event.respondWith(
            caches.open(CACHE_NAME)
                .then((cache) => {
                    return cache.match(event.request)
                        .then((cachedResponse) => {
                            if (cachedResponse) {
                                return cachedResponse;
                            }
                            return fetch(event.request)
                                .then((response) => {
                                    cache.put(event.request, response.clone());
                                    return response;
                                });
                        });
                })
        );
        return;
    }

    // Handle static assets - Cache First strategy
    event.respondWith(
        caches.match(event.request)
            .then((cachedResponse) => {
                if (cachedResponse) {
                    // Return cached version, but update in background
                    event.waitUntil(
                        fetch(event.request)
                            .then((response) => {
                                if (response.status === 200) {
                                    caches.open(CACHE_NAME)
                                        .then((cache) => cache.put(event.request, response));
                                }
                            })
                            .catch(() => {})
                    );
                    return cachedResponse;
                }

                // Not in cache, fetch from network
                return fetch(event.request)
                    .then((response) => {
                        // Don't cache non-successful responses
                        if (!response || response.status !== 200) {
                            return response;
                        }

                        // Clone and cache
                        const responseToCache = response.clone();
                        caches.open(CACHE_NAME)
                            .then((cache) => cache.put(event.request, responseToCache));

                        return response;
                    })
                    .catch(() => {
                        // Return offline page for navigation requests
                        if (event.request.mode === 'navigate') {
                            return caches.match('./index.html');
                        }
                    });
            })
    );
});

// Background Sync for data refresh
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-lotto-data') {
        console.log('[SW] Syncing lotto data...');
        event.waitUntil(
            fetch('https://johannesfriedrich.github.io/LottoNumberArchive/Lottonumbers_tidy_complete.json')
                .then((response) => response.json())
                .then((data) => {
                    return caches.open(DATA_CACHE_NAME)
                        .then((cache) => {
                            return cache.put(
                                'https://johannesfriedrich.github.io/LottoNumberArchive/Lottonumbers_tidy_complete.json',
                                new Response(JSON.stringify(data))
                            );
                        });
                })
                .catch((error) => {
                    console.log('[SW] Sync failed:', error);
                })
        );
    }
});

// Push notifications (for future use)
self.addEventListener('push', (event) => {
    if (event.data) {
        const data = event.data.json();
        const options = {
            body: data.body || 'Neue Lottozahlen verfügbar!',
            icon: './icon-192.png',
            badge: './icon-72.png',
            vibrate: [100, 50, 100],
            data: {
                dateOfArrival: Date.now(),
                primaryKey: '1'
            },
            actions: [
                {
                    action: 'view',
                    title: 'Anzeigen',
                    icon: './icon-72.png'
                },
                {
                    action: 'close',
                    title: 'Schließen'
                }
            ]
        };

        event.waitUntil(
            self.registration.showNotification(data.title || 'LottoGenius', options)
        );
    }
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    if (event.action === 'view' || !event.action) {
        event.waitUntil(
            clients.openWindow('./')
        );
    }
});

// Periodic background sync (if supported)
self.addEventListener('periodicsync', (event) => {
    if (event.tag === 'update-lotto-data') {
        const dataUrl = 'https://johannesfriedrich.github.io/LottoNumberArchive/Lottonumbers_tidy_complete.json';
        event.waitUntil(
            fetch(dataUrl)
                .then((response) => {
                    if (response.ok) {
                        return caches.open(DATA_CACHE_NAME)
                            .then((cache) => cache.put(dataUrl, response));
                    }
                })
        );
    }
});

console.log('[SW] Service Worker loaded');
