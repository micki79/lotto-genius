// 🍀 LottoGenius Service Worker v2.1
// Cache-Version erhöht um alten Cache zu invalidieren!

const CACHE_NAME = 'lotto-genius-v2.1';
const OFFLINE_URL = '/lotto-genius/index.html';

const ASSETS_TO_CACHE = [
    '/lotto-genius/',
    '/lotto-genius/index.html',
    '/lotto-genius/manifest.json',
    '/lotto-genius/icon-72.png',
    '/lotto-genius/icon-96.png',
    '/lotto-genius/icon-128.png',
    '/lotto-genius/icon-144.png',
    '/lotto-genius/icon-152.png',
    '/lotto-genius/icon-192.png',
    '/lotto-genius/icon-384.png',
    '/lotto-genius/icon-512.png',
    '/lotto-genius/icon.svg'
];

// Installation - Cache neue Assets
self.addEventListener('install', (event) => {
    console.log('🍀 Service Worker v2.1 installiert');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('📦 Cache geöffnet');
                return cache.addAll(ASSETS_TO_CACHE);
            })
            .then(() => {
                // Sofort aktivieren ohne auf andere Tabs zu warten
                return self.skipWaiting();
            })
    );
});

// Aktivierung - LÖSCHE ALLE ALTEN CACHES
self.addEventListener('activate', (event) => {
    console.log('🍀 Service Worker v2.1 aktiviert');
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    // Lösche ALLE alten Caches
                    if (cacheName !== CACHE_NAME) {
                        console.log('🗑️ Lösche alten Cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => {
            // Übernehme sofort Kontrolle über alle Clients
            return self.clients.claim();
        })
    );
});

// Fetch - Network First für HTML, Cache First für Assets
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    
    // Ignoriere non-GET requests
    if (event.request.method !== 'GET') {
        return;
    }
    
    // Ignoriere Chrome Extensions und andere Protokolle
    if (!url.protocol.startsWith('http')) {
        return;
    }
    
    // Für HTML-Seiten: Network First (immer aktuelle Version holen)
    if (event.request.mode === 'navigate' || 
        event.request.destination === 'document' ||
        url.pathname.endsWith('.html') ||
        url.pathname.endsWith('/')) {
        
        event.respondWith(
            fetch(event.request)
                .then((response) => {
                    // Erfolgreiche Antwort in Cache speichern
                    if (response.status === 200) {
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(event.request, responseClone);
                        });
                    }
                    return response;
                })
                .catch(() => {
                    // Offline - aus Cache laden
                    return caches.match(event.request)
                        .then((cachedResponse) => {
                            if (cachedResponse) {
                                return cachedResponse;
                            }
                            // Fallback zur Offline-Seite
                            return caches.match(OFFLINE_URL);
                        });
                })
        );
        return;
    }
    
    // Für JSON-Daten: Network First (immer aktuelle Daten holen)
    if (url.pathname.endsWith('.json')) {
        event.respondWith(
            fetch(event.request)
                .then((response) => {
                    if (response.status === 200) {
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(event.request, responseClone);
                        });
                    }
                    return response;
                })
                .catch(() => {
                    return caches.match(event.request);
                })
        );
        return;
    }
    
    // Für statische Assets: Cache First (schneller)
    event.respondWith(
        caches.match(event.request)
            .then((cachedResponse) => {
                if (cachedResponse) {
                    // Im Hintergrund aktualisieren
                    fetch(event.request).then((response) => {
                        if (response.status === 200) {
                            caches.open(CACHE_NAME).then((cache) => {
                                cache.put(event.request, response);
                            });
                        }
                    }).catch(() => {});
                    return cachedResponse;
                }
                
                // Nicht im Cache - vom Netzwerk holen
                return fetch(event.request)
                    .then((response) => {
                        if (response.status === 200) {
                            const responseClone = response.clone();
                            caches.open(CACHE_NAME).then((cache) => {
                                cache.put(event.request, responseClone);
                            });
                        }
                        return response;
                    });
            })
    );
});

// Message Handler - Ermöglicht manuelles Cache-Löschen
self.addEventListener('message', (event) => {
    if (event.data === 'skipWaiting') {
        self.skipWaiting();
    }
    
    if (event.data === 'clearCache') {
        caches.keys().then((cacheNames) => {
            cacheNames.forEach((cacheName) => {
                caches.delete(cacheName);
            });
        });
    }
});

console.log('🍀 LottoGenius Service Worker v2.1 geladen');
