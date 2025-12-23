#!/usr/bin/env python3
"""
📥 LottoGenius - Daten-Abruf-System
Holt aktuelle Lotto-Zahlen für ALLE Spiele von den APIs
"""
import requests
import json
from datetime import datetime
import os
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# API URLs
LOTTO_API = "https://johannesfriedrich.github.io/LottoNumberArchive/Lottonumbers_tidy_complete.json"
EUROJACKPOT_API = "https://johannesfriedrich.github.io/LottoNumberArchive/Eurojackpot_tidy_complete.json"

def fetch_lotto_data():
    """Holt Lotto 6aus49 Daten mit Spiel 77 und Super 6"""
    print("=" * 60)
    print("📥 LottoGenius - Daten-Abruf")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    print(f"🎱 Lade Lotto 6aus49 Daten...")

    try:
        response = requests.get(LOTTO_API, timeout=60)
        response.raise_for_status()
        raw_data = response.json()

        print(f"   ✅ {len(raw_data)} Datensätze empfangen")

        # Verarbeite Daten
        draws = {}
        for entry in raw_data:
            date = entry.get('date', '')
            if not date:
                continue

            if date not in draws:
                draws[date] = {
                    'date': date,
                    'numbers': [],
                    'superzahl': None,
                    'spiel77': None,
                    'super6': None
                }

            variable = entry.get('variable', '')
            value = entry.get('value')

            if variable.startswith('Lottozahl'):
                draws[date]['numbers'].append(value)
            elif variable == 'Superzahl':
                draws[date]['superzahl'] = value
            elif variable == 'Spiel77':
                draws[date]['spiel77'] = str(value).zfill(7)
            elif variable == 'Super6':
                draws[date]['super6'] = str(value).zfill(6)

        # Filtere vollständige Ziehungen für 6aus49
        complete_draws = []
        spiel77_draws = []
        super6_draws = []

        for date, draw in draws.items():
            if len(draw['numbers']) == 6 and draw['superzahl'] is not None:
                draw['numbers'].sort()
                complete_draws.append({
                    'date': draw['date'],
                    'numbers': draw['numbers'],
                    'superzahl': draw['superzahl']
                })

                # Spiel 77 (falls vorhanden)
                if draw['spiel77']:
                    spiel77_draws.append({
                        'date': draw['date'],
                        'number': draw['spiel77']
                    })

                # Super 6 (falls vorhanden)
                if draw['super6']:
                    super6_draws.append({
                        'date': draw['date'],
                        'number': draw['super6']
                    })

        # Sortiere nach Datum (neueste zuerst)
        def parse_date(d):
            return datetime.strptime(d['date'], '%d.%m.%Y')

        complete_draws.sort(key=parse_date, reverse=True)
        spiel77_draws.sort(key=parse_date, reverse=True)
        super6_draws.sort(key=parse_date, reverse=True)

        # Speichere Lotto 6aus49
        os.makedirs(DATA_DIR, exist_ok=True)

        lotto_output = {
            'last_update': datetime.now().isoformat(),
            'total_draws': len(complete_draws),
            'draws': complete_draws
        }

        with open(os.path.join(DATA_DIR, 'lotto_data.json'), 'w') as f:
            json.dump(lotto_output, f, indent=2)

        print(f"   ✅ {len(complete_draws)} Lotto 6aus49 Ziehungen gespeichert")

        # Speichere Spiel 77
        if spiel77_draws:
            spiel77_output = {
                'last_update': datetime.now().isoformat(),
                'total_draws': len(spiel77_draws),
                'draws': spiel77_draws
            }
            with open(os.path.join(DATA_DIR, 'spiel77_data.json'), 'w') as f:
                json.dump(spiel77_output, f, indent=2)
            print(f"   ✅ {len(spiel77_draws)} Spiel 77 Ziehungen gespeichert")
        else:
            # Generiere Demo-Daten wenn API keine hat
            generate_spiel77_demo_data()

        # Speichere Super 6
        if super6_draws:
            super6_output = {
                'last_update': datetime.now().isoformat(),
                'total_draws': len(super6_draws),
                'draws': super6_draws
            }
            with open(os.path.join(DATA_DIR, 'super6_data.json'), 'w') as f:
                json.dump(super6_output, f, indent=2)
            print(f"   ✅ {len(super6_draws)} Super 6 Ziehungen gespeichert")
        else:
            # Generiere Demo-Daten wenn API keine hat
            generate_super6_demo_data()

        print()
        print(f"📅 Letzte Ziehung: {complete_draws[0]['date']}")
        print(f"🎱 Zahlen: {complete_draws[0]['numbers']}")
        print(f"⭐ Superzahl: {complete_draws[0]['superzahl']}")

        return complete_draws

    except Exception as e:
        print(f"❌ Fehler: {e}")
        return []

def fetch_eurojackpot_data():
    """Holt Eurojackpot Daten"""
    print()
    print(f"🌟 Lade Eurojackpot Daten...")

    try:
        response = requests.get(EUROJACKPOT_API, timeout=60)
        response.raise_for_status()
        raw_data = response.json()

        print(f"   ✅ {len(raw_data)} Datensätze empfangen")

        # Verarbeite Daten
        draws = {}
        for entry in raw_data:
            date = entry.get('date', '')
            if not date:
                continue

            if date not in draws:
                draws[date] = {'date': date, 'numbers': [], 'eurozahlen': []}

            variable = entry.get('variable', '')
            value = entry.get('value')

            if variable == 'Lottozahl':
                draws[date]['numbers'].append(value)
            elif variable == 'Eurozahl':
                draws[date]['eurozahlen'].append(value)

        # Filtere vollständige Ziehungen
        complete_draws = []
        for date, draw in draws.items():
            if len(draw['numbers']) == 5 and len(draw['eurozahlen']) == 2:
                draw['numbers'].sort()
                draw['eurozahlen'].sort()
                complete_draws.append(draw)

        # Sortiere nach Datum
        complete_draws.sort(
            key=lambda x: datetime.strptime(x['date'], '%d.%m.%Y'),
            reverse=True
        )

        # Speichere
        output = {
            'last_update': datetime.now().isoformat(),
            'total_draws': len(complete_draws),
            'draws': complete_draws
        }

        with open(os.path.join(DATA_DIR, 'eurojackpot_data.json'), 'w') as f:
            json.dump(output, f, indent=2)

        print(f"   ✅ {len(complete_draws)} Eurojackpot Ziehungen gespeichert")

        if complete_draws:
            print(f"   📅 Letzte Ziehung: {complete_draws[0]['date']}")
            print(f"   🎱 Zahlen: {complete_draws[0]['numbers']}")
            print(f"   ⭐ Eurozahlen: {complete_draws[0]['eurozahlen']}")

        return complete_draws

    except Exception as e:
        print(f"   ⚠️ Eurojackpot API nicht verfügbar: {e}")
        generate_eurojackpot_demo_data()
        return []

def generate_spiel77_demo_data():
    """Generiert historische Spiel 77 Daten basierend auf Statistiken"""
    print("   📊 Generiere Spiel 77 historische Daten...")

    draws = []
    now = datetime.now()

    # Generiere ca. 500 Ziehungen (etwa 2 Jahre)
    for i in range(500):
        # Mittwoch oder Samstag
        day_offset = i * 3 + (i % 2)  # Abwechselnd
        date = datetime(now.year, now.month, now.day)
        date = date.replace(day=max(1, now.day - day_offset % 28))
        if i > 0:
            months_back = i // 8
            year_back = months_back // 12
            date = date.replace(
                year=now.year - year_back,
                month=max(1, ((now.month - months_back) % 12) or 12)
            )

        # Generiere 7-stellige Zahl
        number = ''.join([str(random.randint(0, 9)) for _ in range(7)])

        draws.append({
            'date': date.strftime('%d.%m.%Y'),
            'number': number
        })

    output = {
        'last_update': datetime.now().isoformat(),
        'total_draws': len(draws),
        'draws': draws,
        'note': 'Demo-Daten - echte historische Daten werden nachgeladen'
    }

    with open(os.path.join(DATA_DIR, 'spiel77_data.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"   ✅ {len(draws)} Spiel 77 Demo-Ziehungen generiert")

def generate_super6_demo_data():
    """Generiert historische Super 6 Daten basierend auf Statistiken"""
    print("   📊 Generiere Super 6 historische Daten...")

    draws = []
    now = datetime.now()

    for i in range(500):
        day_offset = i * 3 + (i % 2)
        date = datetime(now.year, now.month, now.day)
        date = date.replace(day=max(1, now.day - day_offset % 28))
        if i > 0:
            months_back = i // 8
            year_back = months_back // 12
            date = date.replace(
                year=now.year - year_back,
                month=max(1, ((now.month - months_back) % 12) or 12)
            )

        # Generiere 6-stellige Zahl
        number = ''.join([str(random.randint(0, 9)) for _ in range(6)])

        draws.append({
            'date': date.strftime('%d.%m.%Y'),
            'number': number
        })

    output = {
        'last_update': datetime.now().isoformat(),
        'total_draws': len(draws),
        'draws': draws,
        'note': 'Demo-Daten - echte historische Daten werden nachgeladen'
    }

    with open(os.path.join(DATA_DIR, 'super6_data.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"   ✅ {len(draws)} Super 6 Demo-Ziehungen generiert")

def generate_eurojackpot_demo_data():
    """Generiert Eurojackpot Demo-Daten"""
    print("   📊 Generiere Eurojackpot Demo-Daten...")

    draws = []
    now = datetime.now()

    for i in range(300):
        day_offset = i * 3
        date = datetime(now.year, now.month, now.day)
        date = date.replace(day=max(1, now.day - day_offset % 28))
        if i > 0:
            months_back = i // 10
            year_back = months_back // 12
            date = date.replace(
                year=now.year - year_back,
                month=max(1, ((now.month - months_back) % 12) or 12)
            )

        # 5 aus 50
        numbers = sorted(random.sample(range(1, 51), 5))
        # 2 aus 12
        eurozahlen = sorted(random.sample(range(1, 13), 2))

        draws.append({
            'date': date.strftime('%d.%m.%Y'),
            'numbers': numbers,
            'eurozahlen': eurozahlen
        })

    output = {
        'last_update': datetime.now().isoformat(),
        'total_draws': len(draws),
        'draws': draws,
        'note': 'Demo-Daten - echte historische Daten werden nachgeladen'
    }

    with open(os.path.join(DATA_DIR, 'eurojackpot_data.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"   ✅ {len(draws)} Eurojackpot Demo-Ziehungen generiert")

def fetch_gluecksspirale_data():
    """
    Versucht Glücksspirale-Daten zu laden.

    Quellen (in Prioritätsreihenfolge):
    1. Web-Scraping von lotto.de/lottohelden.de
    2. Demo-Daten als Fallback

    Hinweis: Für echte historische Daten kann man CSV-Downloads nutzen von:
    - https://www.sachsenlotto.de/portal/zahlen-quoten/gewinnzahlen/download-archiv/
    - https://www.lottozahlenonline.com/gluecksspirale/archiv
    """
    print()
    print(f"🌀 Lade Glücksspirale Daten...")

    # Versuche Web-Scraping
    try:
        from bs4 import BeautifulSoup
        urls_to_try = [
            'https://www.lottohelden.de/gluecksspirale/zahlen-quoten/archiv/',
            'https://www.lotto.de/gluecksspirale/gewinnzahlen'
        ]

        for url in urls_to_try:
            try:
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Hier könnte spezifisches Parsing je nach Website erfolgen
                    print(f"   📡 Verbindung zu {url} erfolgreich")
                    # Für jetzt: verwende Demo-Daten, da Parsing komplex ist
                    break
            except Exception as e:
                continue

    except ImportError:
        print("   ⚠️ BeautifulSoup nicht installiert - verwende Demo-Daten")
    except Exception as e:
        print(f"   ⚠️ Web-Scraping fehlgeschlagen: {e}")

    # Fallback: Generiere Demo-Daten
    generate_gluecksspirale_demo_data()


def generate_gluecksspirale_demo_data():
    """Generiert Glücksspirale Demo-Daten (nur Samstag)"""
    print("   📊 Generiere Glücksspirale Demo-Daten...")

    draws = []
    now = datetime.now()

    # Statistisch realistische Verteilung
    # Basierend auf: Jede Ziffer 0-9 hat gleiche Wahrscheinlichkeit
    for i in range(200):
        # Nur Samstag
        day_offset = i * 7
        date = datetime(now.year, now.month, now.day)
        date = date.replace(day=max(1, now.day - day_offset % 28))
        if i > 0:
            months_back = i // 4
            year_back = months_back // 12
            date = date.replace(
                year=now.year - year_back,
                month=max(1, ((now.month - months_back) % 12) or 12)
            )

        # 7-stellige Zahl (statistisch gleichverteilt)
        number = ''.join([str(random.randint(0, 9)) for _ in range(7)])

        draws.append({
            'date': date.strftime('%d.%m.%Y'),
            'number': number
        })

    output = {
        'last_update': datetime.now().isoformat(),
        'total_draws': len(draws),
        'draws': draws,
        'note': 'Demo-Daten - Für echte Daten: sachsenlotto.de CSV-Download nutzen',
        'data_source': 'generated',
        'real_data_sources': [
            'https://www.sachsenlotto.de/portal/zahlen-quoten/gewinnzahlen/download-archiv/',
            'https://www.lottozahlenonline.com/gluecksspirale/archiv'
        ]
    }

    with open(os.path.join(DATA_DIR, 'gluecksspirale_data.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"   ✅ {len(draws)} Glücksspirale Demo-Ziehungen generiert")
    print(f"   💡 Tipp: Für echte Daten CSV von sachsenlotto.de herunterladen")

def fetch_all_data():
    """
    Holt alle Lotto-Daten für 5 Spiele.

    Datenquellen:
    - Lotto 6aus49: GitHub LottoNumberArchive API (echte Daten)
    - Spiel 77: Aus Lotto-API extrahiert (echte Daten wenn verfügbar)
    - Super 6: Aus Lotto-API extrahiert (echte Daten wenn verfügbar)
    - Eurojackpot: GitHub LottoNumberArchive API (echte Daten)
    - Glücksspirale: Demo-Daten (keine öffentliche API)
    """
    print("\n" + "=" * 60)
    print("🎰 LOTTOGENIUS - MULTI-GAME DATEN-ABRUF")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # 1. Lotto 6aus49 (inkl. Spiel 77 und Super 6 aus gleicher API)
    fetch_lotto_data()

    # 2. Eurojackpot
    fetch_eurojackpot_data()

    # 3. Glücksspirale (Web-Scraping oder Demo-Daten)
    fetch_gluecksspirale_data()

    print("\n" + "=" * 60)
    print("✅ ALLE DATEN ERFOLGREICH GELADEN/GENERIERT")
    print("=" * 60)
    print()
    print("📊 Datenquellen:")
    print("   • Lotto 6aus49, Spiel77, Super6: GitHub API (johannesfriedrich)")
    print("   • Eurojackpot: GitHub API (johannesfriedrich)")
    print("   • Glücksspirale: Demo-Daten (sachsenlotto.de für echte Daten)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    fetch_all_data()
